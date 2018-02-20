#!/bin/bash

function simplify () {
   IN="figs/$1"
   OUT="out/$1"
   python simplify.py --img "$IN" --out "$OUT" || exit 1
}

INPUTS=(
# FIGURE 1
   "fig01_eisaku.png"
)

test -d out/ || mkdir -p out/
for FILE in "${INPUTS[@]}"; do
   echo -n "Processing ${FILE}..."
   simplify "$FILE"
   echo "Done!"
done



