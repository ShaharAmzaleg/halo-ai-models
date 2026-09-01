#!/usr/bin/env python3
# publish.py — מכין פרסום מודל חדש (מריצים רק אחרי שער-איכות שעבר):
#   bump modelVersion בוריאנט "he" ב-manifest.json, מעדכן url/tokenizerUrl/sha256/sizeBytes,
#   וכותב release_tag.txt (models-he-vN) לשימוש ב-gh release.
import hashlib, json

MODEL = "out/model_int8.onnx"
MAN = "manifest.json"
REPO = "ShaharAmzaleg/halo-ai-models"


def main():
    man = json.load(open(MAN, encoding="utf-8"))
    he = next(v for v in man["variants"] if v["id"] == "he")
    newver = int(he.get("modelVersion", 1)) + 1
    tag = f"models-he-v{newver}"
    data = open(MODEL, "rb").read()
    base = f"https://github.com/{REPO}/releases/download/{tag}"
    he["modelVersion"] = newver
    he["url"] = f"{base}/model_int8.onnx"
    he["tokenizerUrl"] = f"{base}/tokenizer.json"
    he["sha256"] = hashlib.sha256(data).hexdigest()
    he["sizeBytes"] = len(data)
    json.dump(man, open(MAN, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    open("release_tag.txt", "w").write(tag)
    print(f"prepared {tag}  sha={he['sha256']}  size={he['sizeBytes']}")


if __name__ == "__main__":
    main()
