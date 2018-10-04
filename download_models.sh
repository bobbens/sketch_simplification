#!/bin/bash

function download_model {
   MODELNAME="$1"
   FILENAME="$2"
   FILEURL="$3"
   FILEMD5="$4"
   echo "Downloading the sketch simplification $MODELNAME model..."
   wget -q --show-progress --continue -O "$FILENAME" -- "$FILEURL"

   echo -n "Checking integrity (md5sum)..."
   OS=`uname -s`
   if [ "$OS" = "Darwin" ]; then
      CHECKSUM=`cat $FILENAME | md5`
   else
      CHECKSUM=`md5sum $FILENAME | awk '{ print $1 }'`
   fi

   if [ "$CHECKSUM" != "$FILEMD5" ]; then
      echo "failed"
      echo "Integrity check failed. File is corrupt!"
      echo "Try running this script again and if it fails remove '$FILENAME' before trying again."
      exit 1
   fi 
   echo "ok"
}

download_model "MSE" "model_mse.t7" "https://esslab.jp/~ess/data/sketch_mse.t7" "12317df9a0a2a7220629f5f361b45b82"
download_model "GAN" "model_gan.t7" "https://esslab.jp/~ess/data/sketch_gan.t7" "3a5b4088f2490ca4b8140a374e80c878"
download_model "PENCIL(1)" "model_pencil1.t7" "https://esslab.jp/~ess/data/pencil_artist1.t7" "33d553ff3a50d6522e79a73002b0025c"
download_model "PENCIL(2)" "model_pencil2.t7" "https://esslab.jp/~ess/data/pencil_artist2.t7" "537b3ad9d46b2a82b65883be747a7ba9"
echo "Downloads finished successfully!"

