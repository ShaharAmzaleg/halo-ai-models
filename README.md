# Halo — AI Models

אירוח מודלים של **זיהוי תוכן פוגעני על-המכשיר** לאפליקציית Halo. ריפו ציבורי (קריאה-בלבד
לעולם), נפרד מקוד האפליקציה. האפליקציה מושכת מכאן את המודל לפי בחירת ההורה, מריצה אותו
**מקומית ואופליין** — הטקסט לעולם לא יוצא מהמכשיר; רק קובץ המודל יורד פעם אחת.

## מבנה
- **`manifest.json`** (מקומט לריפו, נקרא ב-raw) — קטלוג הווריאנטים: לכל ווריאנט `id`, שפות,
  גרסה, כתובת המודל+הטוקנייזר (Release assets), גודל, `sha256`, ו-`toxicIndex`.
- **קבצי המודל** (`*.onnx`, טוקנייזר) — **מתפרסמים כ-GitHub Release assets**, לא מקמיטים לריפו
  (מגבלת git ~100MB; Releases עד 2GB). ראה `.gitignore`.

כתובת ה-manifest שהאפליקציה קוראת:
`https://raw.githubusercontent.com/<YOUR_GITHUB_USER>/halo-ai-models/main/manifest.json`

## הווריאנטים
| id | שפות | מקור |
|----|------|------|
| `multi_all` | ~100 שפות (כולל עברית/אנגלית/רוסית) | `onnx-community/distilbert-multilingual-toxicity-classifier-ONNX` → `model_int8.onnx` (INT8) |
| `he_en` | עברית+אנגלית | *(עתידי — גיזום vocab + כיוונון)* |
| `he_en_ru` | עברית+אנגלית+רוסית | *(עתידי)* |

המודל בינארי: פלט אינדקס **1 = toxic** (0 = not-toxic). ציון = softmax(logits)[1].

## הוספת/עדכון מודל
1. להעלות את קובץ ה-`.onnx` (+טוקנייזר) כ-**Release asset** (tag חדש, למשל `models-v2`).
2. לעדכן את `manifest.json` (כתובת, `sha256`, `sizeBytes`, `modelVersion`).
3. האפליקציה תזהה גרסה חדשה ותוריד — **בלי עדכון אפליקציה**.

מקור המודל הנוכחי: [onnx-community/distilbert-multilingual-toxicity-classifier-ONNX](https://huggingface.co/onnx-community/distilbert-multilingual-toxicity-classifier-ONNX)
· דאטהסט: [textdetox/multilingual_toxicity_dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset) · רישיון המודל: openrail++.
