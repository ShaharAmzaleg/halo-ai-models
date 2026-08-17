#!/usr/bin/env python3
# train_hebrew.py — כיוונון DictaBERT לזיהוי תוכן פוגעני בעברית, וייצוא ל-ONNX INT8 + tokenizer.
# מיועד להרצה ב-Google Colab עם GPU. פלט: out/model_int8.onnx + out/tokenizer.json
#
# הרצה ב-Colab (ראה ההוראות שקיבלת):
#   1) Runtime -> Change runtime type -> GPU (T4)
#   2) !pip install -q "transformers>=4.44" "datasets>=2.20" "optimum[onnxruntime]>=1.21" \
#        onnx onnxruntime evaluate scikit-learn accelerate pandas
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
os.makedirs(OUT, exist_ok=True)


def load_textdetox_he():
    """טוען את פיצול העברית מ-textdetox (מוכן ב-HF). עמודות: text, toxic(0/1)."""
    for kw in [dict(name="he", split="train"), dict(split="he")]:
        try:
            d = load_dataset("textdetox/multilingual_toxicity_dataset", **kw)
            print("textdetox he loaded:", len(d), "| cols:", d.column_names)
            return d
        except Exception as e:
            print("textdetox attempt failed:", e)
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


def build_dataset():
    parts = []
    td = load_textdetox_he()
    if td is not None:
        parts.append(normalize(td))
    if os.path.exists("olah.csv"):
        import pandas as pd
        df = pd.read_csv("olah.csv")
        parts.append(normalize(Dataset.from_pandas(df)))
        print("olah.csv loaded:", len(df))
    if not parts:
        raise RuntimeError("לא נטען אף דאטהסט — ודא רשת/olah.csv")
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    ds = ds.shuffle(seed=42)
    n = len(ds); n_val = max(200, int(n * 0.1))
    print(f"total={n}  train={n-n_val}  val={n_val}  toxic%={100*np.mean(ds['label']):.1f}")
    return ds.select(range(n_val, n)), ds.select(range(n_val))   # train, val


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
                      tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=metrics)
    trainer.train()
    print("FINAL EVAL:", trainer.evaluate())

    # שמירה + ייצוא ONNX + קוונטיזציה INT8
    model.save_pretrained("ft"); tok.save_pretrained("ft")
    from optimum.onnxruntime import ORTModelForSequenceClassification
    ort = ORTModelForSequenceClassification.from_pretrained("ft", export=True)
    ort.save_pretrained("onnx_out")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic("onnx_out/model.onnx", f"{OUT}/model_int8.onnx", weight_type=QuantType.QInt8)
    shutil.copy("ft/tokenizer.json", f"{OUT}/tokenizer.json")

    # בדיקת שפיות מהירה (לפני הורדה)
    print("\n--- sanity check ---")
    for t in ["תודה רבה על העזרה, אתה חבר אמיתי",
              "כולם צוחקים עליך, אתה מביך את כולם",
              "אני מחכה לך אחרי הלימודים ותצטער על זה"]:
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
