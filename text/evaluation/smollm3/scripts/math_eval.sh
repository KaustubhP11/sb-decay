#!/usr/bin/env bash
set -euo pipefail
# Math-only subset from base profile.
TASKS_0SHOT="${TASKS_0SHOT:-}"
TASKS_5SHOT="${TASKS_5SHOT:-gsm8k}"
TASKS_4SHOT="${TASKS_4SHOT:-hendrycks_math}"
export TASKS_0SHOT TASKS_5SHOT TASKS_4SHOT
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/base.sh"
