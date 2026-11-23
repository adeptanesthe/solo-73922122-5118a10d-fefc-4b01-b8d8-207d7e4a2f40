#!/usr/bin/env bash
# Boundary-edit walkthrough against a running service (default
# http://localhost:8000).  It shows how a colour lock excludes the
# highest-scoring boundary:
#
#   1. package A: create + boundary edit -> the highest-scoring boundary
#      hugs the seal column (cuts at 38, touching the seal);
#   2. package B: create + lock the seal to red + the same edit -> the
#      lock makes every seal-touching boundary infeasible, so the service
#      returns the best feasible boundary left of the seal (cuts at 35)
#      with a strictly lower score;
#   3. a stale-version edit is rejected with 409 and the package is kept.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PAGES="$HERE/boundary_pages"
WORK="${WORK:-$HERE/out}"
mkdir -p "$WORK"

if [[ ! -f "$PAGES/manifest.json" ]]; then
  python3 "$HERE/make_boundary_demo_pages.py" "$PAGES" >/dev/null
fi

PALETTE='["#101010","#f2eee2","#a12f26","#2f5d8a","#d8b41e"]'
DISTANCE=200

create_package() {
  curl -fsS -X POST "$BASE/api/packages" \
    -F "rgb_pages=@$PAGES/page1_rgb.png" \
    -F "label_pages=@$PAGES/page1_labels.png" \
    -F "palette=$PALETTE" \
    -F "min_distance=$DISTANCE"
}

boundary_edit() {  # package_id version
  curl -sS -X POST "$BASE/api/packages/$1/boundary-edits" \
    -H 'content-type: application/json' \
    -d "{\"version\": $2, \"page\": 1, \"left_label\": 96, \"right_label\": 32,
         \"rect\": {\"x\": 24, \"y\": 20, \"width\": 16, \"height\": 12}}"
}

echo "== package A: create and re-draw the rule/sky boundary =="
A=$(create_package)
AID=$(echo "$A" | jq -r .package_id)
echo "$A" | jq '{package_id: .package_id, version: .version,
  legend: [.entries[] | {label: .label, colour: .colour}]}'

EDIT_A=$(boundary_edit "$AID" 1)
echo "$EDIT_A" | jq '{version: .version,
  change: {type: .change.type, cuts: .change.cuts, score: .change.score,
           other_label: .change.other_label, search: .change.search}}'
SCORE_A=$(echo "$EDIT_A" | jq -r .change.score)
echo "package A: best boundary scores $SCORE_A (cuts hug the seal column)"

echo
echo "== package B: same page, but the seal (160) is locked to red #a12f26 =="
B=$(create_package)
BID=$(echo "$B" | jq -r .package_id)
LOCKED=$(curl -fsS -X POST "$BASE/api/packages/$BID/locks" \
  -H 'content-type: application/json' -d '{"label": 160, "colour": "#a12f26"}')
echo "$LOCKED" | jq '{version: .version, locks: .locks}'

EDIT_B=$(boundary_edit "$BID" 2)
echo "$EDIT_B" | jq '{version: .version,
  change: {type: .change.type, cuts: .change.cuts, score: .change.score,
           other_label: .change.other_label}}'
SCORE_B=$(echo "$EDIT_B" | jq -r .change.score)

echo
echo "lock effect: highest-scoring boundary excluded by the lock"
echo "  package A (no lock):  score $SCORE_A, cuts at 38 (touch the seal)"
echo "  package B (seal red): score $SCORE_B, cuts at 35 (avoid the seal)"
echo "  red only tolerates the paper colour nearby; the sky already touches"
echo "  the seal and the rule always touches the sky after an edit, so any"
echo "  boundary creating a rule/seal contact is infeasible under the lock."

echo
echo "== revised label map of package B, version 3 (cut between 35 and 36) =="
curl -fsS -o "$WORK/labels-v3.png" "$BASE/api/packages/$BID/versions/3/labels/1"
curl -fsS -o "$WORK/packageB-v3.zip" "$BASE/api/packages/$BID/versions/3/download"
unzip -l "$WORK/packageB-v3.zip" | sed -n '1,12p'

echo
echo "== stale version is rejected and the current package is kept =="
STALE=$(boundary_edit "$BID" 2)
echo "$STALE" | jq '.error | {code: .code, message: .message,
  current_version: .current_version, kept_version: .kept_version}'
echo "package B is still:"
curl -fsS "$BASE/api/packages/$BID" | jq '{version: .version, locks: .locks}'

echo
echo "artifacts in $WORK"
