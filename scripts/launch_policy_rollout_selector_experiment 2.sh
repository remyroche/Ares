#!/usr/bin/env zsh
set -euo pipefail

cd /Users/remyroche/Documents/Ares
exec /bin/zsh /private/tmp/run_policy_rollout_selector_experiment.sh \
  > /Users/remyroche/Documents/Ares/data_perp_policy_rollout_feature_selector_experiment_v3/logs/launchctl.log \
  2>&1
