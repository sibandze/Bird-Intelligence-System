#!/usr/bin/env bash
# filemap - prints a tree-like map from current directory

# usage:./filemap.sh [depth] [path]
# example:./filemap.sh 3.
# example:./filemap.sh # defaults to depth 3, current dir

MAXDEPTH=${1:-3}
ROOT=${2:-.}

print_map() {
  local dir="$1" prefix="$2" depth="$3"
  #(( depth > MAXDEPTH )) && return

  local entries=()
  # nullglob so empty dirs don't break, dotglob to include hidden if you want (remove if not)
  shopt -s nullglob
  mapfile -t entries < <(printf "%s\n" "$dir"/* "$dir"/.* 2>/dev/null | sort -u | grep -v -E "/\.$|/\.\.$")

  local total=${#entries[@]}
  local i=0
  for entry in "${entries[@]}"; do
    ((i++))
    local name=$(basename "$entry")
    local connector="├──"
    local next_prefix="│ "
    (( i == total )) && { connector="└──"; next_prefix=" "; }

    if [ -d "$entry" ] && [ ! -L "$entry" ] && [[ "$entry" != *".git" ]]; then
      echo "${prefix}${connector} ${name}/"
      print_map "$entry" "${prefix}${next_prefix}" $((depth+1))
    else
      echo "${prefix}${connector} ${name}"
    fi
  done
}

echo "${ROOT}/"
print_map "$ROOT" "" 1
