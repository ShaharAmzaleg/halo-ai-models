# פייפליין אימון עם משוב הורים (MLOps)

לולאה: הורה מסמן התראה (פוגעני/לא) → אתה מאשר רשימה → אימון אוטומטי → שער-איכות → פרסום.
הכול רץ ב-GitHub Actions + Firebase (בלי שרת). המודל רץ במכשיר offline; לאימון עובר רק (טקסט, תווית) אנונימי מתוכן שכבר סומן.

## הקמה חד-פעמית
1. **Secret `FIREBASE_SERVICE_ACCOUNT`** בריפו `halo-ai-models` (Settings → Secrets → Actions) — אותו service-account JSON שכבר יש ב-halo-words-updater (קריאה מ-Firestore).
2. **אינדקס Firestore:** הריצה הראשונה של "Collect feedback" עשויה להיכשל עם קישור ליצירת אינדקס ל-collection group `events` על השדה `feedback` — לחץ על הקישור, צור, והרץ שוב.
3. (כבר קיים) `config/ai.updateIntervalDays` שולט בכל כמה זמן המכשיר בודק עדכון מודל.

## המחזור (כל אימון)
1. **הרץ "Collect feedback"** (Actions → Run, או שבועי אוטומטי) → נוצר/מתעדכן `pending_feedback.csv`.
2. **סקור ואשר:** פתח `pending_feedback.csv`. עמודת `decision` **כבר ממולאת** לפי הצעת ההורה — אתה משנה **רק חריגים**:
   - ההורה צדק (רוב המקרים) → **השאר כמו שזה** (מאושר).
   - ההורה טעה → **הפוך** את `decision` (0↔1).
   - לדחות שורה → **רוקן** את התא (decision ריק).
   commit. רק `decision` ∈ {0,1} נכנס לאימון.
3. **הרץ "Train model"** (Actions → Run) → ממזג את המאושרים ל-`feedback.csv` → מאמן → שער-איכות → אם עבר: מפרסם Release + מעדכן manifest, והמכשירים מושכים אוטומטית. אם נכשל: לא מפרסם (נשארים עם הקודם).

## שער-האיכות (evaluate.py)
מפרסם רק אם: precision ≥ 0.90, recall ≥ 0.80, כל `eval/regression_fp.csv` נשאר תמים, ו-F1 ≥ המודל הקודם.
- `eval/eval_set.csv` — סט מבחן נקי (אתה מתחזק). **לא מאמנים עליו.**
- `eval/regression_fp.csv` — טעויות עבר שחייבות להישאר תמימות (לינקים וכו'). הוסף לכאן כל FP חדש.

## קבצים
- `collect_feedback.py` — Firestore → `pending_feedback.csv` (אנונימי, אחרי חלון 7 ימים, מסמן feedbackProcessed).
- `apply_feedback.py` — שורות מאושרות → `feedback.csv`, מרוקן pending.
- `train_hebrew.py` — טוען כל `*.csv` בשורש (כולל `feedback.csv`), מאמן, מייצא ONNX INT8.
- `evaluate.py` — שער-איכות מול `eval/`.
- `publish.py` — bump manifest + tag ל-Release.
- `.github/workflows/feedback.yml` / `train.yml`.

## הערות
- **מחשוב:** רץ על CPU חינמי; אימון DictaBERT על CPU עשוי להימשך זמן (עד ~3ש'). אם איטי מדי — GPU-on-demand.
- **פרטיות:** רק תוכן שכבר סומן (בענן), רק (טקסט, תווית), סקירה אנושית לפני אימון.
