#!/usr/bin/env bash
# Full reproduction: download -> label/split -> train (settings + threshold on validation) -> ONE test evaluation
set -euo pipefail
cd "$(dirname "$0")"
python -m src.download_data
python -m src.prepare_data
python -m src.train
python -m src.evaluate
