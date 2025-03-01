#!/usr/bin/env bash
set -e

cd $(dirname ${0})/..

rm -rf src/turbanet.egg-info
find src -type d -name "__pycache__*" | xargs rm -rf
find ./src -type f -name "*.pyd" -exec rm {} \;

find test -type d -name "__pycache__*" | xargs rm -rf

pip uninstall -y turbanet || true

rm -rf build dist