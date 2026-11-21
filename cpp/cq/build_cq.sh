#!/bin/sh
# Builds the authors' CQ-HUCPM (github.com/hatrasn/High-Utility-Co-location-Pattern-Mining, commit d5dcae8, MIT) on Linux.
#   cq_original : the authors' mining code; only I/O for Linux (command-line arguments, getrusage, no boost CSV split,
#                 MSVC-like FNV-1a hash) is changed  -> cq_linux_original.patch
#   cq_fixed    : the same plus (i) --score total --norm N, the total-utility score of the CQ-HUCPM.py released with
#                 PSO-MUCP, and (ii) the missing 2-cliques: a head with a single star neighbour never enters the
#                 clique-hash table in generateCliques (if (n > 2)), so these pairs are added, and (iii) pattern scores
#                 printed with 9 significant digits (every float exactly), and (iv) the number of patterns whose
#                 utility is computed ("Number of evaluated patterns", I/O only) -> cq_linux_total_fix.patch
# Usage: sh build_cq.sh OUT_DIR
set -e
OUT=${1:-cq_build}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; cd "$OUT"
[ -d src ] || git clone -q https://github.com/hatrasn/High-Utility-Co-location-Pattern-Mining.git src
git -C src checkout -q d5dcae87ad7f2d99c66eae9c187bf231e04a72c3
for v in original total_fix; do
  rm -rf "$v"; cp -r src/CQ-HUCPM "$v"; (cd "$v" && patch -s -p1 < "$HERE/cq_linux_$v.patch")
  g++ -O2 -std=c++14 -o "$v/cq_hucpm" "$v"/CQ-HUCPM.cpp "$v"/Object.cpp "$v"/printFunction.cpp "$v"/readerCSV.cpp "$v"/toolFunctions.cpp
done
echo built "$OUT/original/cq_hucpm" "$OUT/total_fix/cq_hucpm"
