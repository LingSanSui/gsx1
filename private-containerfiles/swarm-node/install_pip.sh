#!/bin/bash

set +euo pipefail
# shellcheck disable=SC1090
source ~/.profile
set -euo pipefail

# Install our packages
pip install  --user -r ./requirements-hivemind.txt
pip install  --user -r ./requirements.txt