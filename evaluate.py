#!/usr/bin/env python3
# evaluate.py — שער-איכות. מריץ את המודל שאומן (out/model_int8.onnx) על סטי מבחן נקיים
# ומחזיר קוד יציאה 0 (עבר) רק אם עמד בכל התנאים; אחרת 1 (נכשל) → הפייפליין לא מפרסם.
#
# תנאים:
#   precision >= MIN_PRECISION  (מעט false-positive)
#   recall    >= MIN_RECALL     (לא מפספס יותר מדי)
#   כל השורות ב-regression_fp.csv נשארות תמימות (0 שגיאות שחזרו)
#   F1 >= baseline (אם קיים eval/baseline_f1.txt מהמודל הקודם) — לא יורדים
#
# ההסקה כאן מיישרת קו עם המכשיר: אותו ניקוי (URLs/@/#) ואותה טוקניזציה (WordPiece מ-tokenizer.json).

import csv, os, re, sys, unicodedata
import numpy as np
import onnxruntime as ort
import json

THRESHOLD = 0.8
MIN_PRECISION = 0.90
MIN_RECALL = 0.80
MODEL = "out/model_int8.onnx"
TOK = "out/tokenizer.json"
EVAL = "eval/eval_set.csv"
REG = "eval/regression_fp.csv"
BASELINE = "eval/baseline_f1.txt"


def load_tokenizer(path):
    d = json.load(open(path, encoding="utf-8"))
    m = d["model"]; vocab = m["vocab"]
    prefix = m.get("continuing_subword_prefix", "##")
    maxchars = m.get("max_input_chars_per_word", 100)
    norm = d.get("normalizer", {}) or {}
    lower = norm.get("lowercase", False)
    sa = norm.get("strip_accents")
    sa = lower if sa is None else sa
    unk, cls, sep = vocab["[UNK]"], vocab["[CLS]"], vocab["[SEP]"]

    def clean(t):
        t = re.sub(r"https?://\S+", " ", str(t)); t = re.sub(r"@\w+", " ", t)
        t = re.sub(r"\bRT\b:?", " ", t); t = t.replace("#", " ")
        return re.sub(r"\s+", " ", t).strip()

    def strip_accents(s):
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    def is_punct(c):
        cp = ord(c)
        if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
            return True
        return unicodedata.category(c).startswith("P")

    def basic(text):
        if sa: text = strip_accents(text)
        if lower: text = text.lower()
        out = []
        for tok in text.split():
            cur = ""
            for c in tok:
                if is_punct(c):
                    if cur: out.append(cur); cur = ""
                    out.append(c)
                else: cur += c
            if cur: out.append(cur)
        return out

    def wp(w):
        if len(w) > maxchars: return [unk]
        st, sub = 0, []
        while st < len(w):
            en, cur = len(w), None
            while st < en:
                piece = (prefix + w[st:en]) if st > 0 else w[st:en]
                if piece in vocab: cur = vocab[piece]; break
                en -= 1
            if cur is None: return [unk]
            sub.append(cur); st = en
        return sub

    def encode(text):
        text = clean(text)
        ids = [cls]
        for w in basic(text): ids += wp(w)
        return ids[:127] + [sep], (len(clean(text)) == 0)

    return encode


def main():
    enc = load_tokenizer(TOK)
    sess = ort.InferenceSession(MODEL)
    names = [i.name for i in sess.get_inputs()]

    def score(text):
        ids, empty = enc(text)
        if empty: return 0.0        # אחרי ניקוי אין טקסט (לינק בלבד) → תמים
        a = np.array([ids], dtype=np.int64); mask = np.ones_like(a); tt = np.zeros_like(a)
        feed = {"input_ids": a, "attention_mask": mask}
        if "token_type_ids" in names: feed["token_type_ids"] = tt
        lg = sess.run(None, feed)[0][0]
        e = np.exp(lg - lg.max()); return float((e / e.sum())[1])

    # eval_set → precision/recall/F1
    tp = fp = fn = tn = 0
    with open(EVAL, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y = int(r["label"]); pred = 1 if score(r["text"]) >= THRESHOLD else 0
            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 1: fn += 1
            else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # regression_fp → כולם חייבים לצאת תמימים
    reg_fail = []
    if os.path.exists(REG):
        with open(REG, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if score(r["text"]) >= THRESHOLD:
                    reg_fail.append(r["text"])

    baseline = None
    if os.path.exists(BASELINE):
        try: baseline = float(open(BASELINE).read().strip())
        except Exception: baseline = None

    print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f} "
          f"reg_fail={len(reg_fail)} baseline_F1={baseline}")
    for t in reg_fail:
        print(f"  REGRESSION FP still flagged: {t}")

    ok = (precision >= MIN_PRECISION and recall >= MIN_RECALL and not reg_fail
          and (baseline is None or f1 >= baseline - 0.005))
    if ok:
        with open("eval/new_f1.txt", "w") as f: f.write(f"{f1:.4f}")   # לקידום ל-baseline אחרי פרסום
        print("QUALITY GATE: PASS ✅"); sys.exit(0)
    print("QUALITY GATE: FAIL ❌ — not publishing"); sys.exit(1)


if __name__ == "__main__":
    main()
