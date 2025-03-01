#!/usr/bin/env bash
set -e

cd $(dirname ${0})/..

python -m build --sdist
python -m build --wheel

pip uninstall -y turbanet || true
pip install --no-index dist/turbanet-*.whl