#!/bin/zsh
# Bounded monitor for the side-local representation search.  It exits when the
# search exits and never changes model/data artifacts.
set -euo pipefail

out_dir="${1:?output directory required}"
interval_seconds="${2:-300}"
log_file="$out_dir/run_monitor.log"

while true; do
  now="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  pids="$(pgrep -f 'run_side_local_ae_gmm_representation_search.py' || true)"
  {
    print -- "[$now] pids=${pids:-none}"
    if [[ -n "$pids" ]]; then
      pid_list="${pids//$'\n'/,}"
      ps -p "$pid_list" -o pid=,stat=,etime=,%cpu=,%mem=,rss=,command= || true
    fi
    print -- "prepare_manifests=$(find "$out_dir" -name prepare_manifest.json -type f | wc -l | tr -d ' ') proxy_reports=$(find "$out_dir" -name proxy_candidates.csv -type f | wc -l | tr -d ' ') full_reports=$(find "$out_dir" -name full_candidates.csv -type f | wc -l | tr -d ' ')"
  } >> "$log_file"
  [[ -n "$pids" ]] || exit 0
  sleep "$interval_seconds"
done
