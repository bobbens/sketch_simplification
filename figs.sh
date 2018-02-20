#!/bin/bash

function simplify () {
   IN="figs/$1"
   OUT="out/$1"
   python simplify.py --img "$IN" --out "$OUT" || exit 1
}

INPUTS=(
# FIGURE 1
   "fig01_eisaku.png"
# FIGURE 6
   "fig06_eisaku_joshi.png"
   "fig06_pepper.png"
   "fig06_danshi.png"
   "fig06_eisaku_robo.png"
# FIGURE 7
   "fig07_tokage.png"
# FIGURE 12
   "fig12_imori.png"
   "fig12_shikaku.png"
   "fig12_pepper.png"
# FIGURE 14
   "fig14_imori.png"
   "fig14_origami.png"
   "fig14_pepper.png"
   "fig14_danshi.png"
# FIGURE 15
   "fig15_kame.png"
   "fig15_joshi.png"
# FIGURE 16
   "fig16_eisaku.png"
)

test -d out/ || mkdir -p out/
for FILE in "${INPUTS[@]}"; do
   echo -n "Processing ${FILE}..."
   simplify "$FILE"
   echo "Done!"
done



