#!/usr/bin/env python3
# apply_feedback.py — ממזג את השורות ש**אישרת** ב-pending_feedback.csv אל feedback.csv (דאטת אימון),
# ואז מרוקן את pending (משאיר כותרת בלבד). מריצים לפני האימון.
#
# רק שורות עם decision ∈ {0,1} נכנסות (decision הוא התווית הסופית שקבעת — מאשר או מתקן).
# שורות עם decision ריק = נדחו ולא נכנסות. dedup מול feedback.csv הקיים.

import csv, os

PENDING = "pending_feedback.csv"
FEEDBACK = "feedback.csv"


def main():
    if not os.path.exists(PENDING):
        print("no pending file — nothing to apply"); return

    approved = []
    with open(PENDING, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dec = (r.get("decision") or "").strip()
            text = (r.get("text") or "").strip()
            if dec in ("0", "1") and text:
                approved.append((text, dec))

    existing, rows = set(), []
    if os.path.exists(FEEDBACK):
        with open(FEEDBACK, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                t = (r.get("text") or "").strip(); l = (r.get("label") or "").strip()
                if t and (t, l) not in existing:
                    existing.add((t, l)); rows.append((t, l))

    added = 0
    for text, label in approved:
        if (text, label) not in existing:
            existing.add((text, label)); rows.append((text, label)); added += 1

    with open(FEEDBACK, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["text", "label"])
        for t, l in rows:
            w.writerow([t, l])

    with open(PENDING, "w", encoding="utf-8", newline="") as f:      # מרוקן את הממתינים
        csv.writer(f).writerow(["text", "suggested_label", "decision"])

    print(f"applied {added} new rows -> {FEEDBACK} (total {len(rows)})")


if __name__ == "__main__":
    main()
