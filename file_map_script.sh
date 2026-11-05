#!/usr/bin/env bash
# filemap - prints a full tree-like map from current directory
# with git last-updated date per file
#
# usage: ./file_map_script.sh [path] [--all] [--no-git]
# example: ./file_map_script.sh
# example: ./file_map_script.sh src
# example: ./file_map_script.sh . --all
# example: ./file_map_script.sh . --no-git

ROOT="."
SHOW_ALL=0
SHOW_GIT=1

# Parse args
for arg in "$@"; do
  if [[ "$arg" == "--all" ]]; then
    SHOW_ALL=1
  elif [[ "$arg" == "--no-git" ]]; then
    SHOW_GIT=0
  else
    ROOT="$arg"
  fi
done

# Files/folders to ignore in the tree output
IGNORE_PATTERNS=(
  ".vscode"
  ".venv"
  "*cache"
  ".git"
  "__pycache__"
  "*.pyc"
  "*_review.md"
  "*.egg-info"
  ".DS_Store"
  "node_modules"
  "*.ogg"
  "*.npy"
)

# Cache: is this a git repo?
IN_GIT=0
if (( SHOW_GIT == 1 )) && git rev-parse --git-dir >/dev/null 2>&1; then
  IN_GIT=1
fi

git_date() {
  local file="$1"
  (( IN_GIT == 0 )) && return

  # Last commit touching this file: YYYY-MM-DD
  local date
  date=$(git log -1 --format=%ci -- "$file" 2>/dev/null)

  if [[ -z "$date" ]]; then
    # No commit yet - check if tracked at all
    if git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
      echo -n " [tracked, no commit?]"
    else
      echo -n " [untracked]"
    fi
    return
  fi

  # * = modified since last commit (dirty)
  local dirty=""
  if ! git diff --quiet -- "$file" 2>/dev/null; then
    dirty="*"
  elif ! git diff --cached --quiet -- "$file" 2>/dev/null; then
    dirty="*"
  fi

  echo -n " [$date$dirty]"
}

print_map() {
  local dir="$1" prefix="$2"

  local entries=()
  shopt -s nullglob dotglob
  mapfile -t entries < <(printf "%s\n" "$dir"/* "$dir"/.* 2>/dev/null | sort -u | grep -v -E "/\.$|/\.\.$")

  local filtered=()
  for entry in "${entries[@]}"; do
    local name=$(basename "$entry")
    local skip=0
    if (( SHOW_ALL == 0 )); then
      for pat in "${IGNORE_PATTERNS[@]}"; do
        [[ "$name" == $pat ]] && skip=1 && break
      done
    fi
    (( skip == 0 )) && filtered+=("$entry")
  done

  local total=${#filtered[@]}
  local i=0
  for entry in "${filtered[@]}"; do
    ((i++))
    local name=$(basename "$entry")
    local connector="├──"
    local next_prefix="│ "
    (( i == total )) && { connector="└──"; next_prefix=" "; }

    if [ -d "$entry" ] && [ ! -L "$entry" ]; then
      echo "${prefix}${connector} ${name}/"
      print_map "$entry" "${prefix}${next_prefix}"
    else
      local suffix=""
      (( SHOW_GIT == 1 )) && suffix=$(git_date "$entry")
      echo "${prefix}${connector} ${name}${suffix}"
    fi
  done
}

if (( SHOW_ALL == 1 )); then
  echo "${ROOT}/ [ALL FILES]"
else
  echo "${ROOT}/"
fi
print_map "$ROOT" ""
