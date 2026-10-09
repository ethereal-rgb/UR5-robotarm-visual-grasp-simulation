#!/usr/bin/env bash
# Export the reviewed source files without nested repositories or local state.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export_dir="${1:-$(mktemp -d /tmp/robot_nocatch_github.XXXXXX)}"
if [[ -e "$export_dir" && ! -d "$export_dir" ]]; then
  echo "目标必须是目录：$export_dir" >&2
  exit 1
fi
mkdir -p -- "$export_dir"
export_dir="$(cd -- "$export_dir" && pwd)"
if [[ -n "$(ls -A -- "$export_dir")" ]]; then
  echo "目标目录必须为空：$export_dir" >&2
  exit 1
fi
tar -C "$project_root" --exclude='.git' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='src/CMakeLists.txt' \
  -cf - README.md .gitignore docs tools src | tar -C "$export_dir" -xf -
printf '源码已导出：%s\n' "$export_dir"
