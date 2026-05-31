#!/usr/bin/env bash
# Symlink the plugin into HermesAgent's USER plugin dir (outside the repo → git-pull-safe).
set -euo pipefail
SRC="$HOME/projects/step37-harness/plugin"
DST="$HOME/.hermes/plugins/fleet-orchestrator"
mkdir -p "$HOME/.hermes/plugins"
ln -sfn "$SRC" "$DST"
echo "symlinked  $DST  ->  $SRC"
echo
echo "enable it with either:"
echo "  hermes plugins enable fleet-orchestrator"
echo "  # or add 'fleet-orchestrator' under plugins.enabled in ~/.hermes/config.yaml"
