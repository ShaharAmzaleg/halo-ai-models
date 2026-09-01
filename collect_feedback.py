#!/usr/bin/env python3
# collect_feedback.py — אוסף משובי הורים מ-Firestore ומייצר קובץ סקירה pending_feedback.csv.
# רץ ב-GitHub Actions עם Firebase Admin SDK (סוד FIREBASE_SERVICE_ACCOUNT).
#
# פרטיות: מחלץ אך ורק (טקסט, תווית) אנונימי — בלי שם שולח/קבוצה/משפחה/ילד/זמן.
# נאסף רק תוכן ש**כבר סומן** (כבר בענן, ההורה ראה אותו), ורק אחרי ש**חלון העריכה של 7 ימים נסגר**
# (התווית סופית). כל אירוע שנאסף מסומן feedbackProcessed=true כדי שלא ייאסף שוב (הנתון נשמר ב-CSV/גיט).
#
# הפלט pending_feedback.csv: עמודות text, suggested_label, decision.
#   suggested_label = מה שההורה סימן (1=פוגעני, 0=לא). decision = ריק (ממתין לאישורך):
#   מלא 1/0 כדי לאשר/לתקן, השאר ריק כדי לדחות. רק decision ∈ {0,1} נכנס לאימון (apply_feedback.py).

import csv, json, os, re, time
import firebase_admin
from firebase_admin import credentials, firestore

# חלון עריכת המשוב — אוספים רק אחרי שנסגר (התווית סופית). ברירת מחדל 7 ימים;
# ניתן לעקוף להרצת-בדיקה ע"י FEEDBACK_MIN_AGE_DAYS=0 (input ב-workflow).
WINDOW_MS = int(os.environ.get("FEEDBACK_MIN_AGE_DAYS", "7")) * 24 * 3600 * 1000
OUT = "pending_feedback.csv"


def clean(t):
    """ניקוי זהה ל-clean_tweet/ההסקה: URLs/@/RT/#, כיווץ רווחים."""
    t = str(t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"\bRT\b:?", " ", t)
    t = t.replace("#", " ")
    return re.sub(r"\s+", " ", t).strip()


def main():
    sa = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    if not sa:
        raise SystemExit("FIREBASE_SERVICE_ACCOUNT missing")
    firebase_admin.initialize_app(credentials.Certificate(json.loads(sa)))
    db = firestore.client()
    now = int(time.time() * 1000)

    rows, to_mark, seen = [], [], set()
    # collection group query על כל אוספי "events" בכל המשפחות/ילדים
    q = db.collection_group("events").where("feedback", "in", ["tp", "fp"])
    for doc in q.stream():
        d = doc.to_dict() or {}
        fb = d.get("feedback")
        if fb not in ("tp", "fp") or d.get("feedbackProcessed"):
            continue
        ts = d.get("timestamp") or 0
        if now - ts < WINDOW_MS:          # חלון עדיין פתוח → התווית עוד יכולה להשתנות
            continue
        to_mark.append(doc.reference)      # נאסף (או ידולג) → לא ייאסף שוב
        text = clean(d.get("messageText", ""))
        label = 1 if fb == "tp" else 0
        if len(text) < 2 or (text, label) in seen:
            continue
        seen.add((text, label))
        rows.append((text, label))

    # צבירה (לא דריסה): קוראים את ה-pending הקיים כדי לא לאבד משוב משבוע קודם שטרם עובד,
    # ומוסיפים רק שורות חדשות (dedup לפי טקסט). עריכות decision קיימות נשמרות.
    # (אחרי אימון, apply_feedback.py מרוקן את pending → ה-collect הבא מתחיל נקי.)
    existing, existing_texts = [], set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if len(row) >= 3 and row[0].strip():
                    existing.append((row[0], row[1], row[2])); existing_texts.add(row[0].strip())

    added = 0
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "suggested_label", "decision"])
        for t, sl, dec in existing:                       # קודם הקיימים (עם העריכות שלך)
            w.writerow([t, sl, dec])
        for text, label in rows:                          # ואז החדשים בלבד
            if text.strip() in existing_texts:
                continue
            # decision ממולא מראש לפי הצעת ההורה — אתה משנה רק חריגים (הפוך 0↔1 / רוקן לדחייה).
            w.writerow([text, label, label]); added += 1
    print(f"appended {added} new rows (kept {len(existing)} existing) -> {OUT}")

    # סימון כמעובד (בbatches של 400 — מגבלת Firestore)
    batch, n = db.batch(), 0
    for ref in to_mark:
        batch.update(ref, {"feedbackProcessed": True}); n += 1
        if n % 400 == 0:
            batch.commit(); batch = db.batch()
    if n % 400 != 0:
        batch.commit()
    print(f"marked {n} events processed")


if __name__ == "__main__":
    main()
