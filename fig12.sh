#!/bin/bash

function simplify () {
   IN="figs/$1"
   OUT="out/$1"
   python simplify.py --img "$IN" --out "$OUT" || exit 1
}

INPUTS=(
# FIGURE 12
   "fig12_imori.png"
   "fig12_shikaku.png"
   "fig12_pepper.png"
)

test -d out/ || mkdir -p out/
for FILE in "${INPUTS[@]}"; do
   echo -n "Processing ${FILE}..."
   simplify "$FILE"
   echo "Done!"
done



