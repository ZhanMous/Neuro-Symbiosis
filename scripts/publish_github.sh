#!/usr/bin/env bash
set -euo pipefail

REPO_NAME=${1:-Neuro-Symbiosis}
VISIBILITY=${2:-public}

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install GitHub CLI first."
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub auth not found. Run: gh auth login"
  exit 2
fi

if [ ! -d .git ]; then
  git init -b main
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "feat: initialize Neuro-Symbiosis project with experiments and draft paper"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source . --remote origin --push
else
  git push -u origin main
fi

echo "Publish completed."
