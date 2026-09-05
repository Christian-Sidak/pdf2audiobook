#!/bin/bash
# Assemble the Spader build past the 3 reviewed residual flags, embed cover, push to iCloud.
cd "$(dirname "$0")/.."
BOOK=how_to_win_friends_and_influence_people_rev_ed
LOG=artifacts/$BOOK/render_spader_v2.log
echo "=== finish: stages 5-6 --keep-going $(date '+%F %T')" >> "$LOG"
caffeinate -is python3 -u main.py build "library/How to Win Friends and Influence People (rev ed).pdf" \
  --stages 5-6 --engine qwen3tts --voice voices/james_spader.wav \
  --title "How to Win Friends and Influence People" --author "Dale Carnegie" --keep-going >> "$LOG" 2>&1
rc=$?; echo "=== finish exited rc=$rc $(date '+%F %T')" >> "$LOG"
if [ $rc -eq 0 ]; then
  python3 main.py cover $BOOK artifacts/$BOOK/cover_spader.png >> "$LOG" 2>&1
  python3 -c "from export.icloud import publish; publish('$BOOK')" >> "$LOG" 2>&1
  echo "SPADER V2 BUILD DONE $(date '+%F %T')" | tee -a "$LOG"
else
  echo "SPADER V2 FINISH FAILED rc=$rc" | tee -a "$LOG"
fi
