#!/usr/bin/env bash
set -euo pipefail

# clean_snapshot.sh

# Remove any .pkl file that is not model_best.pkl
# Remove any 2nd-level folder that doesn't have a model_best.pkl under it

DRYRUN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run|-n] [SNAPSHOT_DIR]

If SNAPSHOT_DIR is not provided, defaults to 'snapshot'.

Options:
  -n, --dry-run   Show what would be deleted without removing anything
  -h, --help      Show this help message
EOF
}

while [[ ${1-} =~ ^- ]]; do
  case "$1" in
    -n|--dry-run) DRYRUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

TARGET_DIR="${1:-snapshot}"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Directory not found: $TARGET_DIR" >&2
  exit 1
fi

echo "Operating on: $TARGET_DIR"
if [ "$DRYRUN" -eq 1 ]; then
  echo "DRY RUN: no files will be deleted"
fi

# 1) Find .pkl files that are not model_best.pkl and remove them (everywhere)
echo "Finding .pkl files (excluding model_best.pkl)..."
found_any=0
while IFS= read -r -d '' file; do
  found_any=1
  if [ "$DRYRUN" -eq 1 ]; then
    printf "Would remove file: %s\n" "$file"
  else
    printf "Removing file: %s\n" "$file"
    rm -f -- "$file"
  fi
done < <(find "$TARGET_DIR" -type f -iname '*.pkl' ! -name 'model_best.pkl' -print0)

if [ "$found_any" -eq 0 ]; then
  echo "No matching .pkl files found."
fi

# Only consider immediate child directories (experiment folders) of the target
# For each child dir, if it does NOT contain a model_best.pkl anywhere in its
# subtree, remove the entire child dir. Do not remove the snapshot root itself.
echo "Removing immediate child directories under $TARGET_DIR that do NOT contain model_best.pkl (dry-run=$DRYRUN)..."
found_removed=0
while IFS= read -r -d '' child; do
  # check if model_best.pkl exists anywhere inside this child directory
  if find "$child" -type f -name 'model_best.pkl' -print -quit | grep -q .; then
    # contains model_best.pkl, skip
    continue
  fi
  found_removed=1
  if [ "$DRYRUN" -eq 1 ]; then
    printf "Would remove experiment directory (no model_best.pkl): %s\n" "$child"
  else
    printf "Removing experiment directory (no model_best.pkl): %s\n" "$child"
    rm -rf -- "$child"
  fi
done < <(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -print0)

if [ "$found_removed" -eq 0 ]; then
  echo "No immediate child directories without model_best.pkl found."
fi

echo "Done."
