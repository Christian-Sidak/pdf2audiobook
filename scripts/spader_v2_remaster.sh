#!/bin/bash
# Reassemble the Spader master with the lower limiter ceiling, re-embed cover, push.
cd "$(dirname "$0")/.."
BOOK=how_to_win_friends_and_influence_people_rev_ed
LOG=artifacts/$BOOK/render_spader_v2.log
echo "=== remaster: limiter 0.5 $(date '+%F %T')" >> "$LOG"
caffeinate -is python3 -u main.py reassemble $BOOK >> "$LOG" 2>&1 || { echo "REMASTER FAILED" | tee -a "$LOG"; exit 1; }
python3 main.py cover $BOOK artifacts/$BOOK/cover_spader.png >> "$LOG" 2>&1
python3 -c "from export.icloud import publish; publish('$BOOK')" >> "$LOG" 2>&1
echo "SPADER V2 REMASTER DONE $(date '+%F %T')" | tee -a "$LOG"
