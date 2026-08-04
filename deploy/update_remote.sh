#!/usr/bin/env bash
# Ship the current working tree to a deployed server and restart the service.
#
#   ./deploy/update_remote.sh user@host              # deploy
#   ./deploy/update_remote.sh -n user@host           # dry run: show what would change
#   REMOTE_DIR=~/xaikit-api ./deploy/update_remote.sh user@host
#
# Runs stage_deploy.sh first, so a tree carrying participant data is never sent.
#
# What is deliberately NOT synced, and why --delete would otherwise destroy it:
# the staged tree holds only src/, server/, assets/ and requirements.txt, so
# rsync counts everything else on the server as extraneous. Without these
# excludes a deploy removes the .env holding the API token and empties
# server_runs/.
#
# Restarting drops every in-memory study: a study is a live xaikitTest object,
# and only files already written under server_runs/ survive. Study ids from
# before the restart return 404, so deploy when nobody is mid-run.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE_DIR:-/tmp/xaikit-deploy}"
REMOTE_DIR="${REMOTE_DIR:-~/xaikit-api}"
SERVICE="${SERVICE:-xaikit-api}"
DRY=""

while getopts "nh" opt; do
  case "$opt" in
    n) DRY="--dry-run" ;;
    h) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))

TARGET="${1:-${REMOTE_HOST:-}}"
if [ -z "$TARGET" ]; then
  echo "usage: $0 [-n] user@host" >&2
  exit 2
fi

echo "==> Staging a tree with participant data removed"
"$REPO/deploy/stage_deploy.sh" "$STAGE" >/dev/null
echo "    $(du -sh "$STAGE" | cut -f1) at $STAGE"

echo "==> Syncing to $TARGET:$REMOTE_DIR ${DRY:+(dry run)}"
# --delete keeps the remote tree from accumulating files deleted locally; the
# excludes are what stop it deleting server-owned state. Never drop them.
rsync -a --delete $DRY --itemize-changes \
  --exclude='.env' \
  --exclude='server_runs/' \
  --exclude='deploy/' \
  "$STAGE/" "$TARGET:$REMOTE_DIR/"

if [ -n "$DRY" ]; then
  echo "==> Dry run only. Nothing was changed and the service was not restarted."
  exit 0
fi

echo "==> Restarting $SERVICE (in-memory studies are dropped here)"
ssh "$TARGET" "systemctl --user restart $SERVICE && sleep 3 && systemctl --user is-active $SERVICE"

echo "==> Health check"
if ssh "$TARGET" "curl -fsS localhost:8000/api/health"; then
  echo
  echo "==> Deployed."
else
  echo >&2
  echo "Health check failed. Recent log:" >&2
  ssh "$TARGET" "journalctl --user -u $SERVICE -n 30 --no-pager" >&2
  exit 1
fi
