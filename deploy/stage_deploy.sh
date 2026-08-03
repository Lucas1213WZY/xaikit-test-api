#!/usr/bin/env bash
# Build a copy of this repo with every human-participant path removed, ready to
# ship to a server that is not running Docker.
#
# .dockerignore protects Docker builds only. A git clone or an rsync of the repo
# carries assets/human_data/ and src/cognitive_models/CoAX/results/ with it, and
# on a shared machine other accounts may be able to read them. Deploy from this
# staged tree instead of from the repo.
#
#   ./deploy/stage_deploy.sh /tmp/xaikit-deploy
#   rsync -av --delete /tmp/xaikit-deploy/ user@server:~/xaikit-api/
#
# The excluded paths are the same list as .dockerignore. Keep the two in step.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-/tmp/xaikit-deploy}"

# Human participant data and reference-study material the service never reads.
SRC_EXCLUDES=(
  'cognitive_models/CoAX/results/'
  'cognitive_models/CoAX/data/'
  'cognitive_models/CoAX/UI/'
  'cognitive_models/CoAX/simulation_mockup/'
  'cognitive_models/CoXAM/datasets/'
  'cognitive_models/CoXAM/outputs/'
)
ASSET_EXCLUDES=(
  'human_data/'
  'human_trials_and_cognitive_parameters/'
)

rm -rf "$STAGE"
mkdir -p "$STAGE"

src_args=()
for pattern in "${SRC_EXCLUDES[@]}"; do src_args+=(--exclude="$pattern"); done
asset_args=()
for pattern in "${ASSET_EXCLUDES[@]}"; do asset_args+=(--exclude="$pattern"); done

common=(--exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store')

rsync -a "${common[@]}" "${src_args[@]}" "$REPO/src/" "$STAGE/src/"
rsync -a "${common[@]}" "${asset_args[@]}" "$REPO/assets/" "$STAGE/assets/"
rsync -a "${common[@]}" "$REPO/server/" "$STAGE/server/"
cp "$REPO/requirements.txt" "$STAGE/"

echo "Staged $(du -sh "$STAGE" | cut -f1) at $STAGE"

# Fail loudly rather than shipping a tree that still holds participant records.
leaked="$(find "$STAGE" \
  \( -path '*human_data*' -o -path '*CoAX/results*' -o -path '*human_trials*' \) \
  -print | head)"
if [ -n "$leaked" ]; then
  echo "REFUSING: human data present in the staged tree:" >&2
  echo "$leaked" >&2
  exit 1
fi
echo "Verified: no human participant data in the staged tree."
