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

WINDOW_MS = 7 * 24 * 3600 * 1000   # חלון עריכת המשוב — אוספים רק אחרי שנסגר
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

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "suggested_label", "decision"])
        for text, label in rows:
            w.writerow([text, label, ""])
    print(f"wrote {len(rows)} pending rows -> {OUT}")

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
