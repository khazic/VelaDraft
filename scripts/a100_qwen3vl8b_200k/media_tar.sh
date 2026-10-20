#!/bin/bash
# Ship a run's node-local images to other nodes through the shared disk (large files only):
#
#   bash scripts/a100_qwen3vl8b_200k/media_tar.sh pack    <images_dir> <tar_dir>
#   bash scripts/a100_qwen3vl8b_200k/media_tar.sh restore <tar_dir> <images_dir> [<regen run dir>]
#
# pack writes 16 tar shards + files.txt + sha256 of every shard (a few big sequential
# files on the shared disk instead of hundreds of thousands of small ones); restore
# checks the shards, unpacks them into the node-local dir, checks the file count and,
# with a run dir, that every image its data/regen rows reference exists.
# Prints MEDIA_PACKED / MEDIA_OK.
set -Eeuo pipefail
trap 'echo "MEDIA_ABORT line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR
mode=${1:?pack|restore}
if [ "$mode" = pack ]; then
  src=${2:?images dir} dst=${3:?tar dir}
  mkdir -p "$dst"
  ( cd "$src" && ls ) > "$dst/files.txt"
  split -n r/16 -d "$dst/files.txt" "$dst/list_"
  for list in "$dst"/list_*; do tar cf "$dst/part_${list##*_}.tar" -C "$src" -T "$list" & done
  wait
  ( cd "$dst" && sha256sum part_*.tar > checksums.txt )
  echo "MEDIA_PACKED files=$(wc -l < "$dst/files.txt") shards=$(ls "$dst"/part_*.tar | wc -l)"
else
  src=${2:?tar dir} dst=${3:?images dir} run=${4:-}
  ( cd "$src" && sha256sum -c checksums.txt --quiet )
  mkdir -p "$dst"
  ls "$src"/part_*.tar | xargs -P 8 -I{} tar xf {} -C "$dst"
  have=$(ls "$dst" | wc -l) want=$(wc -l < "$src/files.txt")
  [ "$have" -ge "$want" ] || { echo "MEDIA_FAIL count $have < $want"; exit 1; }
  if [ -n "$run" ]; then
    python3 - "$run" "$dst" <<'PY'
import json, os, sys
from pathlib import Path

run, dst = Path(sys.argv[1]), sys.argv[2]
missing = rows = 0
for path in [run / "regen" / "target_data.jsonl", *sorted((run / "data").glob("*.jsonl"))]:
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        images = row.get("images") or ([row["image"]] if row.get("image") else [])
        rows += 1
        missing += sum(not os.path.exists(os.path.join(dst, name)) for name in images)
print(f"checked {rows} rows, missing images: {missing}")
sys.exit(1 if missing else 0)
PY
  fi
  echo "MEDIA_OK files=$have"
fi
