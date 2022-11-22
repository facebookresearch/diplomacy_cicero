#!/bin/bash
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
set -eu

CWD=$(pwd)
BINDIR=$(dirname $(realpath $0))
ROOT=$(dirname $BINDIR)
OUTDIR=$ROOT/models
INDIR=$ROOT/models_encrypted

PASSWORD=$1


###
# Guard statements to confirm input is well-formed.
###
if [ -d $OUTDIR ]
then
    echo "$OUTDIR already exists. Not going to risk overwriting it. Quitting..."
    exit 0
fi

if [ -d $INDIR ]
then
    echo "$INDIR already exists. Not going to risk overwriting it. Quitting..."
    exit 0
fi

PWD_LENGTH=${#PASSWORD}
if [ ! $PWD_LENGTH -eq 30 ]
then
    echo "You either did not provide a password, or it was the wrong length."
    exit 0
fi

####
# Downloading models_encrypted
####
mkdir $INDIR
FILENAMES=$BINDIR/s3_filenames.txt

cat $FILENAMES | while read FILE
do
    echo $FILE
    wget https://dl.fbaipublicfiles.com/diplomacy_cicero/models/$FILE.gpg -O $INDIR/$FILE.gpg
done


####
# Running commands to unencrypt models_encrypted
####

mkdir $OUTDIR
mkdir $OUTDIR/nonsense_ensemble

cd $INDIR

for FILE in *;
do
    if [ "$FILE" != "nonsense_ensemble" ]; then
        STEM=$(echo $FILE | rev | cut -c5- | rev)
        OUTFILE=$OUTDIR/$STEM
        gpg --batch --yes --passphrase $PASSWORD --output $OUTFILE -d $FILE;
    fi
done;

cd nonsense_ensemble

for FILE in *;
do
    STEM=$(echo $FILE | rev | cut -c5- | rev)
    OUTFILE=$OUTDIR/nonsense_ensemble/$STEM
    gpg --batch --yes --passphrase $PASSWORD --output $OUTFILE -d $FILE;
done;

cd $CWD


