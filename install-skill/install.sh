#!/usr/bin/env bash
set -euo pipefail

# Usage: invoke this skill with the repository name (e.g., "anthropics/pdf")
# It clones the repo into ~/.clawd/skills and makes it available.

REPO="${1:-anthropics/pdf}"
BASE_DIR="$HOME/.clawd/skills"
REPO_DIR="$BASE_DIR/$(echo "$REPO" | tr '/' '-')"

if [ -d "$REPO_DIR" ]; then
  echo "Skill '$REPO' is already installed at $REPO_DIR"
  exit 0
fi

echo "Cloning skill '$REPO' into $BASE_DIR ..."
git clone "https://github.com/$REPO.git" "$REPO_DIR"

echo "Skill installed successfully at $REPO_DIR"
echo "You can now use it via Clawd‑Code."