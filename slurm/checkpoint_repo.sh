#!/bin/bash
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
set -e

if [[ $# -lt 1 ]]
then
    BASEDIR=~/diplomacy_experiments/repo_checkpoints/
else
    BASEDIR="$1"
fi

mkdir -p "$BASEDIR"
TMPDIR=$(mktemp -d -p "$BASEDIR")
ROOT="$(realpath $(dirname $0)/..)"
TARBALL=repo.tar

echo "Syncing $ROOT --> $TMPDIR" 1>&2
tar -C "$ROOT" -cf "$TMPDIR"/$TARBALL \
    --exclude-vcs \
    --exclude .git \
    --exclude '*.db' \
    --exclude '*.pt' \
    --exclude 'nohup.viz.log' \
    --exclude build \
    --exclude fairdiplomacy/viz/web/node_modules \
    --exclude wandb \
    --exclude dipcc \
    --exclude unit_tests \
    --exclude src \
    --exclude thirdparty/github/fairinternal/postman \
    --exclude .mypy_cache \
    --exclude local \
    --exclude tmp \
    --exclude models \
    --exclude build \
    --exclude wandb \
    .

tar -C "$TMPDIR" -xf "$TMPDIR"/$TARBALL
rm "$TMPDIR"/$TARBALL

(git rev-parse HEAD > "$TMPDIR"/GITHASH.txt) || true
(git diff HEAD > "$TMPDIR"/GITDIFF.txt) || true

echo "$TMPDIR"
