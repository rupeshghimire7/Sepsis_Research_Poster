#!/bin/sh
# Render poster.html to an A0 PDF with headless Chrome (and a small PNG preview for checking).
cd "$(dirname "$0")"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT=${1:-Poster_SepsisSense_Sepsis_Risk_ED_Triage.pdf}
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=3000 \
  --print-to-pdf="$OUT" "file://$PWD/poster.html" 2>/dev/null
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=0.5 \
  --window-size=3179,4494 --virtual-time-budget=3000 --screenshot=preview.png "file://$PWD/poster.html" 2>/dev/null
echo "wrote $OUT and preview.png"
