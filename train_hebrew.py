#!/usr/bin/env python3
# train_hebrew.py — כיוונון DictaBERT לזיהוי תוכן פוגעני בעברית, וייצוא ל-ONNX INT8 + tokenizer.
# מיועד להרצה ב-Google Colab עם GPU. פלט: out/model_int8.onnx + out/tokenizer.json
#
# הרצה ב-Colab (ראה ההוראות שקיבלת):
#   1) Runtime -> Change runtime type -> GPU (T4)
#   2) !pip install -q "transformers>=4.44" "datasets>=2.20" onnx onnxruntime onnxscript evaluate scikit-learn accelerate pandas
#   3) (אופציונלי) להעלות olah.csv עם עמודות text,label (label: 1=פוגעני, 0=תמים)
#   4) !python train_hebrew.py

import os, json, hashlib, shutil
import numpy as np
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import f1_score, accuracy_score

BASE   = "dicta-il/dictabert"     # BERT עברי סטנדרטי (WordPiece — תואם למנוע באפליקציה)
MAXLEN = 128
OUT    = "out"
LANGS  = ["he", "en"]             # שפות מ-textdetox לטעון (עברית = ליבה; אנגלית = כיסוי נוסף). רוסית בהמשך.
MAX_PER_LANG = 3000               # תקרה לכל שפה — שומר על איזון כדי שהעברית לא תיבלע.
# OLaH — קורפוס עברי אמיתי (16k ציוצים, SinaLab). דאטה אמיתי מגוון עם פיסוק טבעי → מתקן שבריריות.
OLAH_URL = "https://raw.githubusercontent.com/SinaLab/OffensiveHebrew/main/data/AllData_OffensiveHebrew.csv"
MAX_OLAH_BENIGN = 6000            # תקרת תמימים מ-OLaH — הרבה עברית אמיתית תמימה, בלי להבליע את שאר הדאטה.
os.makedirs(OUT, exist_ok=True)


def clean_tweet(t):
    """מנקה ארטיפקטים של טוויטר (URLs, @אזכורים, RT, #) כדי שהמודל לא ילמד תבניות ספציפיות לטוויטר."""
    import re
    t = str(t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"\bRT\b:?", " ", t)
    t = t.replace("#", " ")
    return re.sub(r"\s+", " ", t).strip()


def load_olah():
    """טוען את OLaH מ-GitHub (TweetText + Label). מיפוי בינארי: NOT→0, כל שאר (Hate/Violence/Abusive/racism/Porn)→1.
       מנקה ציוצים ומגביל את מספר התמימים ל-MAX_OLAH_BENIGN לשמירת איזון."""
    import pandas as pd
    try:
        df = pd.read_csv(OLAH_URL)
    except Exception as e:
        print("OLaH load failed:", e); return None
    if "TweetText" not in df.columns or "Label" not in df.columns:
        print("OLaH: unexpected columns", list(df.columns)); return None
    texts, labels = [], []
    for _, row in df.iterrows():
        lab = str(row["Label"]).strip()
        if lab == "" or lab.lower() == "nan":   # ללא תווית → מדלגים (לא מנחשים)
            continue
        y = 0 if lab == "NOT" else 1
        txt = clean_tweet(row["TweetText"])
        if len(txt) < 3:
            continue
        texts.append(txt); labels.append(y)
    d = Dataset.from_dict({"text": texts, "label": labels})
    ben = d.filter(lambda r: r["label"] == 0)
    tox = d.filter(lambda r: r["label"] == 1)
    if len(ben) > MAX_OLAH_BENIGN:
        ben = ben.shuffle(seed=42).select(range(MAX_OLAH_BENIGN))
    d = concatenate_datasets([ben, tox])
    print(f"OLaH loaded: benign={len(ben)} toxic={len(tox)}")
    return d


def load_textdetox(lang):
    """טוען פיצול שפה מ-textdetox (מוכן ב-HF). עמודות: text, toxic(0/1). מוגבל ל-MAX_PER_LANG."""
    for kw in [dict(name=lang, split="train"), dict(split=lang)]:
        try:
            d = load_dataset("textdetox/multilingual_toxicity_dataset", **kw)
            if len(d) > MAX_PER_LANG:
                d = d.shuffle(seed=42).select(range(MAX_PER_LANG))
            print(f"textdetox {lang} loaded:", len(d), "| cols:", d.column_names)
            return d
        except Exception as e:
            print(f"textdetox {lang} attempt failed:", e)
    return None


def normalize(d):
    """מנרמל ל-שתי עמודות: text (str), label (int 0/1)."""
    cols = d.column_names
    label_col = "toxic" if "toxic" in cols else ("label" if "label" in cols else None)
    text_col = "text" if "text" in cols else cols[0]
    d = d.rename_columns({text_col: "text"}) if text_col != "text" else d
    if label_col and label_col != "label":
        d = d.rename_columns({label_col: "label"})
    d = d.select_columns(["text", "label"])
    d = d.map(lambda r: {"text": str(r["text"]).strip(), "label": int(r["label"])})
    return d.filter(lambda r: len(r["text"]) > 0)


def augment_punctuation(ds):
    """לכל דוגמה מוסיף וריאנט ללא פיסוק סופי (?/!/./פסיק/רווח), כדי לנתק את הפיסוק מהתווית.
       מונע שהמודל ילמד קיצור-דרך כמו 'סימן שאלה = תמים'. מוחל על train בלבד."""
    import re
    texts, labels = ds["text"], ds["label"]
    add_t, add_l = [], []
    for t, l in zip(texts, labels):
        s = re.sub(r"[\s\.\?!,،؛:]+$", "", t)
        if s and s != t:
            add_t.append(s); add_l.append(l)
    if not add_t:
        return ds
    print(f"punctuation-augment: +{len(add_t)} variants (ללא פיסוק סופי)")
    return concatenate_datasets([ds, Dataset.from_dict({"text": add_t, "label": add_l})])


def build_dataset():
    import glob, pandas as pd
    parts = []
    olah = load_olah()                 # דאטה עברי אמיתי (OLaH) — הבסיס לתיקון השבריריות
    if olah is not None:
        parts.append(normalize(olah))
    for lang in LANGS:
        td = load_textdetox(lang)
        if td is not None:
            parts.append(normalize(td))
    # טוען כל קובץ CSV מקומי עם עמודות text,label (he_augment.csv, olah.csv, וכו')
    for path in sorted(glob.glob("*.csv")):
        try:
            df = pd.read_csv(path)
            if "text" in df.columns and "label" in df.columns:
                parts.append(normalize(Dataset.from_pandas(df)))
                print(f"{path} loaded:", len(df))
        except Exception as e:
            print(f"skip {path}:", e)
    if not parts:
        raise RuntimeError("לא נטען אף דאטהסט — ודא רשת/קבצי CSV")
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    ds = ds.shuffle(seed=42)
    n = len(ds); n_val = max(200, int(n * 0.1))
    print(f"total={n}  train={n-n_val}  val={n_val}  toxic%={100*np.mean(ds['label']):.1f}")
    val = ds.select(range(n_val))                       # ולידציה — נקייה (בלי אוגמנטציה)
    train = augment_punctuation(ds.select(range(n_val, n))).shuffle(seed=42)   # train + וריאנטים ללא פיסוק
    return train, val


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, id2label={0: "not-toxic", 1: "toxic"}, label2id={"not-toxic": 0, "toxic": 1})

    train_ds, val_ds = build_dataset()

    def tok_fn(b): return tok(b["text"], truncation=True, max_length=MAXLEN)
    train_ds = train_ds.map(tok_fn, batched=True)
    val_ds = val_ds.map(tok_fn, batched=True)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"f1": f1_score(p.label_ids, preds), "acc": accuracy_score(p.label_ids, preds)}

    args = TrainingArguments(
        output_dir="ckpt", num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="no", logging_steps=50,
        fp16=torch.cuda.is_available(), report_to="none")

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=metrics)
    trainer.train()
    print("FINAL EVAL:", trainer.evaluate())

    # שמירה + ייצוא ONNX (torch.onnx — בלי optimum) + קוונטיזציה INT8
    model.save_pretrained("ft"); tok.save_pretrained("ft")
    model.eval().to("cpu")
    import torch.nn as nn
    class Wrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits
    dummy = tok("שלום עולם", return_tensors="pt", truncation=True, max_length=MAXLEN)
    os.makedirs("onnx_out", exist_ok=True)
    torch.onnx.export(
        Wrap(model),
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        "onnx_out/model.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                      "token_type_ids": {0: "b", 1: "s"}, "logits": {0: "b"}},
        opset_version=14, dynamo=False)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic("onnx_out/model.onnx", f"{OUT}/model_int8.onnx", weight_type=QuantType.QInt8)
    shutil.copy("ft/tokenizer.json", f"{OUT}/tokenizer.json")

    # בדיקת שפיות מהירה (לפני הורדה)
    print("\n--- sanity check ---")
    for t in ["תודה רבה על העזרה, אתה חבר אמיתי",
              "כולם צוחקים עליך, אתה מביך את כולם",
              "אני מחכה לך אחרי הלימודים ותצטער על זה",
              "מה קורה איתך?",           # עם סימן שאלה — היה תקין
              "מה קורה איתך",            # בלי סימן שאלה — היה 88% (שבריריות פיסוק) → אמור לרדת
              "מה איתך",                 # בלי סימן שאלה
              "בוא נלך לשחק עכשיו בפארק",  # תמים עם מילים שדלפו
              "See you tomorrow, thanks for the help",
              "You're pathetic and everyone laughs at you",
              "I'll make you regret coming to school"]:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=MAXLEN)
        with torch.no_grad():
            prob = torch.softmax(model.to("cpu")(**enc).logits, -1)[0, 1].item()
        print(f"{prob*100:5.1f}%  {t}")

    # גודל + sha256 (ל-manifest)
    size = os.path.getsize(f"{OUT}/model_int8.onnx")
    sha = hashlib.sha256(open(f"{OUT}/model_int8.onnx", "rb").read()).hexdigest()
    print(f"\nmodel_int8.onnx  size={size}  sha256={sha}")

    try:
        from google.colab import files
        files.download(f"{OUT}/model_int8.onnx"); files.download(f"{OUT}/tokenizer.json")
    except Exception:
        print(f"\nfiles ready in ./{OUT}/")


if __name__ == "__main__":
    main()
