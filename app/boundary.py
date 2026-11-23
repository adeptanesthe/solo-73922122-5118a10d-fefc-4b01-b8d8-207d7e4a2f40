"""Boundary re-drawing between two labels inside a small rectangle.

When the boundary between two numbered regions of a guide page was drawn
in the wrong place, an editor submits the package id, the current version,
the page number, the left and right label values and a rectangle of at most
24x24 pixels.  Inside that rectangle:

* every row picks a cut point between two adjacent *editable* pixels
  (pixels currently carrying the left or right label);
* cut points of neighbouring rows may differ by at most one pixel;
* only pixels carrying the two labels are re-assigned: left of the cut
  becomes the left label, right of the cut becomes the right label;
* the rectangle plus the outside pixels sharing an edge with it may
  contain at most one other label.

The score of a boundary is the sum, over the rectangle rows, of the
squared RGB distance between the two original pixels flanking the cut.
Among the boundaries that keep the whole set colourable under the original
minimum colour distance and the active locks, the highest score wins; ties
are broken by comparing cut points row by row from the top, preferring the
cut that is further left first (lexicographically smallest cut vector).

Why the search does not enumerate all boundaries
------------------------------------------------
With up to 23 cut positions per row and 24 rows there are O(23**24) cut
vectors.  Two structural facts make an exact dynamic program possible:

1. The score is separable per row and the smoothness constraint couples
   only adjacent rows, so without the colourability constraint the problem
   is a plain row-by-row DP.

2. Colourability depends on the whole cut vector only through a two-bit
   *contact mask*.  Because the rectangle neighbourhood contains no labels
   other than left/right and at most one other label ``O``, re-labelling
   can only create or remove contacts among ``{L, R, O}``.  The cut itself
   always joins L and R, so the only vector-dependent contacts are
   ``L-O`` and ``R-O``.  Feasibility of the recolouring (graph + palette
   distance + locks) is independent of pixel costs, so each of the four
   masks is checked once with the exact colouring solver and the DP keeps
   the mask in its state.

The DP state is ``(row, cut, mask)`` — at most 24 * 24 * 4 states — and
keeps the best ``(score, cut vector)`` prefix per state, ordered by score
descending and then by the top-down left-most tie-break.  Dominance is
sound: two prefixes reaching the same state with equal score keep only the
lexicographically smaller one, because appending any common suffix
preserves both the score order and the tie-break.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .imaging import RGB, InputError, Page
from .pipeline import Analysis
from .solver import solve_colouring

MAX_RECT_SIDE = 24

# Contact-mask bits: bit 0 = left/other contact, bit 1 = right/other contact.
LEFT_OTHER = 1
RIGHT_OTHER = 2


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def check_inside(self, page_width: int, page_height: int) -> None:
        if not (1 <= self.width <= MAX_RECT_SIDE):
            raise InputError(
                f"rect width must be between 1 and {MAX_RECT_SIDE}, got {self.width}"
            )
        if not (1 <= self.height <= MAX_RECT_SIDE):
            raise InputError(
                f"rect height must be between 1 and {MAX_RECT_SIDE}, got {self.height}"
            )
        if (
            self.x < 0
            or self.y < 0
            or self.x + self.width > page_width
            or self.y + self.height > page_height
        ):
            raise InputError(
                f"rect (x={self.x}, y={self.y}, {self.width}x{self.height}) lies "
                f"outside the {page_width}x{page_height} page"
            )


@dataclass
class RowPlan:
    """Per-row search data: allowed cuts, their scores and contact masks."""

    y: int
    cuts: list[int]
    scores: dict[int, int]
    masks: dict[int, int]


@dataclass
class BoundaryPlan:
    """Validated, immutable input for the boundary search."""

    page_index: int
    left: int
    right: int
    other: int | None
    rect: Rect
    rows: list[RowPlan]
    # Pixels subject to re-labelling: rect pixels labelled left/right.
    changeable: frozenset[tuple[int, int]]


@dataclass
class BoundarySolution:
    cuts: list[int]
    score: int
    # Search-effort telemetry proving the cut vectors were not enumerated.
    dp_states: int
    cut_vectors: int
    feasible_masks: list[int]


def plan_boundary_edit(
    page: Page, page_index: int, left: int, right: int, rect: Rect
) -> BoundaryPlan:
    """Validate the rectangle and precompute per-row cut data.

    Raises ``InputError`` when the rectangle cannot host a boundary edit.
    """
    rect.check_inside(page.width, page.height)
    labels = page.labels
    width = page.width
    height = page.height
    rgb = page.rgb
    pair = {left, right}

    # Labels inside the rect plus the ring of outside pixels sharing an
    # edge with it may introduce at most one other label.
    neighbourhood: set[int] = set()
    changeable: set[tuple[int, int]] = set()
    for y in range(rect.y, rect.y + rect.height):
        base = y * width
        for x in range(rect.x, rect.x + rect.width):
            value = labels[base + x]
            neighbourhood.add(value)
            if value in pair:
                changeable.add((y, x))
    for x in range(rect.x, rect.x + rect.width):
        if rect.y > 0:
            neighbourhood.add(labels[(rect.y - 1) * width + x])
        if rect.y + rect.height < height:
            neighbourhood.add(labels[(rect.y + rect.height) * width + x])
    for y in range(rect.y, rect.y + rect.height):
        if rect.x > 0:
            neighbourhood.add(labels[y * width + rect.x - 1])
        if rect.x + rect.width < width:
            neighbourhood.add(labels[y * width + rect.x + rect.width])

    others = sorted(neighbourhood - pair)
    if len(others) > 1:
        raise InputError(
            "the rectangle and the pixels sharing an edge with it contain "
            f"more than one other label: {others}"
        )
    other = others[0] if others else None

    rows: list[RowPlan] = []
    for y in range(rect.y, rect.y + rect.height):
        base = y * width
        cuts: list[int] = []
        scores: dict[int, int] = {}
        masks: dict[int, int] = {}
        for c in range(rect.x + 1, rect.x + rect.width):
            # The cut sits between pixels c-1 and c; both must be editable.
            if labels[base + c - 1] not in pair or labels[base + c] not in pair:
                continue
            cuts.append(c)
            first = (base + c - 1) * 3
            second = (base + c) * 3
            dr = rgb[first] - rgb[second]
            dg = rgb[first + 1] - rgb[second + 1]
            db = rgb[first + 2] - rgb[second + 2]
            scores[c] = dr * dr + dg * dg + db * db
            masks[c] = _row_mask(labels, width, height, rect, y, c, pair, other)
        if not cuts:
            raise InputError(
                f"row {y} of the rectangle has no adjacent pair of pixels "
                f"labelled {left} or {right} to place a cut between"
            )
        rows.append(RowPlan(y=y, cuts=cuts, scores=scores, masks=masks))

    return BoundaryPlan(
        page_index=page_index,
        left=left,
        right=right,
        other=other,
        rect=rect,
        rows=rows,
        changeable=frozenset(changeable),
    )


def _row_mask(
    labels: bytes,
    width: int,
    height: int,
    rect: Rect,
    y: int,
    cut: int,
    pair: set[int],
    other: int | None,
) -> int:
    """Which left/right contacts with the other label this cut creates.

    Only pixels of row ``y`` are re-labelled by this row's cut, and every
    neighbour of an editable pixel lies inside the rect or its edge ring,
    so the other label is the only contact partner that can appear.
    """
    if other is None:
        return 0
    mask = 0
    base = y * width
    for x in range(rect.x, rect.x + rect.width):
        if labels[base + x] not in pair:
            continue
        bit = LEFT_OTHER if x < cut else RIGHT_OTHER
        if x > 0 and labels[base + x - 1] == other:
            mask |= bit
        if x + 1 < width and labels[base + x + 1] == other:
            mask |= bit
        if y > 0 and labels[base - width + x] == other:
            mask |= bit
        if y + 1 < height and labels[base + width + x] == other:
            mask |= bit
    return mask


def residual_edges(
    pages: Sequence[Page],
    edited_page_index: int,
    changeable: frozenset[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Label contacts of the whole set that no cut vector can change.

    A contact is vector-dependent only when one of its pixels is editable;
    every other contact survives the edit and forms the residual graph.
    """
    edges: set[tuple[int, int]] = set()
    for page_index, page in enumerate(pages):
        labels = page.labels
        width = page.width
        height = page.height
        edited = page_index == edited_page_index
        for y in range(height):
            base = y * width
            for x in range(width):
                a = labels[base + x]
                if x + 1 < width:
                    b = labels[base + x + 1]
                    if a != b and not (
                        edited
                        and ((y, x) in changeable or (y, x + 1) in changeable)
                    ):
                        edges.add((a, b) if a < b else (b, a))
                if y + 1 < height:
                    b = labels[base + width + x]
                    if a != b and not (
                        edited
                        and ((y, x) in changeable or (y + 1, x) in changeable)
                    ):
                        edges.add((a, b) if a < b else (b, a))
    return edges


def solve_boundary(
    pages: Sequence[Page],
    analysis: Analysis,
    palette: Sequence[RGB],
    min_distance: float,
    locks: dict[int, int],
    plan: BoundaryPlan,
) -> BoundarySolution | None:
    """Best-scoring feasible cut vector, or None when none is feasible.

    Runs at most four exact colourability checks (one per contact mask)
    and a row-by-row dynamic program; it never enumerates the cut vectors.
    """
    residual = residual_edges(pages, plan.page_index, plan.changeable)
    labels = analysis.labels
    index = {label: i for i, label in enumerate(labels)}
    costs = [analysis.costs[label] for label in labels]
    locked_vertices = {index[label]: colour for label, colour in locks.items()}

    def norm(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    left_right = norm(plan.left, plan.right)
    masks = [0] if plan.other is None else [0, 1, 2, 3]
    feasible: dict[int, bool] = {}
    for mask in masks:
        edges = set(residual)
        # The cut always makes the two labels touch.
        edges.add(left_right)
        if plan.other is not None:
            if mask & LEFT_OTHER:
                edges.add(norm(plan.left, plan.other))
            if mask & RIGHT_OTHER:
                edges.add(norm(plan.right, plan.other))
        vertex_edges = [(index[a], index[b]) for a, b in edges]
        result = solve_colouring(
            costs, vertex_edges, palette, min_distance, locked_vertices
        )
        feasible[mask] = result.assignment is not None

    # Dynamic program over (cut, contact mask) per row.  Each state keeps
    # the best (score, cut-vector) prefix; ties prefer the vector whose cut
    # is further left in the earliest differing row.
    states = 0
    first = plan.rows[0]
    previous: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
    for c in first.cuts:
        key = (c, first.masks[c])
        candidate = (first.scores[c], (c,))
        if key not in previous or _better(candidate, previous[key]):
            previous[key] = candidate
    states += len(previous)

    for row in plan.rows[1:]:
        current: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
        for (prev_cut, prev_mask), (prev_score, prev_cuts) in previous.items():
            for c in row.cuts:
                if abs(c - prev_cut) > 1:
                    continue
                key = (c, prev_mask | row.masks[c])
                candidate = (prev_score + row.scores[c], prev_cuts + (c,))
                if key not in current or _better(candidate, current[key]):
                    current[key] = candidate
        previous = current
        states += len(previous)

    best: tuple[int, tuple[int, ...]] | None = None
    for (cut, mask), (score, cuts) in previous.items():
        if not feasible[mask]:
            continue
        candidate = (score, cuts)
        if best is None or _better(candidate, best):
            best = candidate
    if best is None:
        return None

    vectors = 1
    for row in plan.rows:
        vectors *= len(row.cuts)
    return BoundarySolution(
        cuts=list(best[1]),
        score=best[0],
        dp_states=states,
        cut_vectors=vectors,
        feasible_masks=sorted(mask for mask, ok in feasible.items() if ok),
    )


def _better(
    a: tuple[int, tuple[int, ...]], b: tuple[int, tuple[int, ...]]
) -> bool:
    """Higher score wins; ties prefer the top-down left-most cut vector."""
    return a[0] > b[0] or (a[0] == b[0] and a[1] < b[1])


def apply_cuts(page: Page, plan: BoundaryPlan, cuts: Sequence[int]) -> bytes:
    """Re-label the editable pixels of ``page`` according to the cuts."""
    data = bytearray(page.labels)
    pair = {plan.left, plan.right}
    for row, cut in zip(plan.rows, cuts):
        base = row.y * page.width
        for x in range(plan.rect.x, plan.rect.x + plan.rect.width):
            if data[base + x] in pair:
                data[base + x] = plan.left if x < cut else plan.right
    return bytes(data)
