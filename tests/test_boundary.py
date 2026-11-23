from __future__ import annotations

import io
import json
import random
import zipfile
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.boundary import (
    Rect,
    apply_cuts,
    plan_boundary_edit,
    solve_boundary,
)
from app.imaging import InputError, Page
from app.main import app, store
from app.pipeline import analyse_pages
from app.solver import solve_colouring
from app.store import PackageStore, VersionConflictError


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_page(index, width, height, label_rows, rgb_rows) -> Page:
    rgb = bytearray()
    for row in rgb_rows:
        for pixel in row:
            rgb.extend(pixel)
    labels = bytes(value for row in label_rows for value in row)
    return Page(
        index=index, width=width, height=height, rgb=bytes(rgb), labels=labels
    )


def upload_page(rgb_rows, label_rows):
    height = len(rgb_rows)
    width = len(rgb_rows[0])
    rgb = Image.new("RGB", (width, height))
    rgb.putdata([tuple(px) for row in rgb_rows for px in row])
    labels = Image.new("L", (width, height))
    labels.putdata([value for row in label_rows for value in row])
    return (
        [("rgb_pages", ("page1.png", png_bytes(rgb), "image/png"))],
        [("label_pages", ("page1-labels.png", png_bytes(labels), "image/png"))],
    )


def submit(client, rgb_files, label_files, palette, min_distance, locks=None):
    data = {"palette": json.dumps(palette), "min_distance": str(min_distance)}
    if locks is not None:
        data["locks"] = json.dumps(locks)
    return client.post("/api/packages", data=data, files=rgb_files + label_files)


# --------------------------------------------------------------------------
# Demo page: a grey rule band (96) left of a blue sky band (32), with a
# one-column red seal (160) inside the sky band.  The edit rectangle
# straddles the rule/sky boundary; the highest-contrast original pixels sit
# just right of the seal column, so the unconstrained best boundary hugs
# the seal and creates a rule/seal contact.
# --------------------------------------------------------------------------
PALETTE5 = ["#101010", "#f2eee2", "#a12f26", "#2f5d8a", "#d8b41e"]
PALETTE5_RGB = [
    (16, 16, 16),
    (242, 238, 226),
    (161, 47, 38),
    (47, 93, 138),
    (216, 180, 30),
]
PAPER, RULE, SKY, SEAL = 240, 96, 32, 160
RECT = {"x": 24, "y": 20, "width": 16, "height": 12}


def boundary_demo_rows():
    width = height = 64
    labels = [[PAPER] * width for _ in range(height)]
    rgb = [[(242, 238, 226)] * width for _ in range(height)]
    for y in range(16, 49):
        for x in range(16, 29):
            labels[y][x] = RULE
            rgb[y][x] = (150, 140, 120)
        for x in range(29, 45):
            labels[y][x] = SKY
            rgb[y][x] = (47, 93, 138)
    for y in range(20, 32):
        labels[y][36] = SEAL
        rgb[y][36] = (168, 58, 44)
        rgb[y][35] = (20, 20, 20)
        rgb[y][37] = (10, 10, 10)
        rgb[y][38] = (250, 250, 250)
        rgb[y][39] = (10, 10, 10)
    return rgb, labels


def create_demo_package(client, locks=None):
    rgb_rows, label_rows = boundary_demo_rows()
    rgb_files, label_files = upload_page(rgb_rows, label_rows)
    response = submit(client, rgb_files, label_files, PALETTE5, 200, locks=locks)
    assert response.status_code == 201, response.text
    return response.json()


def edit_body(version):
    return {
        "version": version,
        "page": 1,
        "left_label": RULE,
        "right_label": SKY,
        "rect": RECT,
    }


def test_boundary_edit_success(client: TestClient) -> None:
    created = create_demo_package(client)
    package_id = created["package_id"]

    response = client.post(
        f"/api/packages/{package_id}/boundary-edits", json=edit_body(1)
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["version"] == 2
    change = body["change"]
    assert change["type"] == "boundary_edit"
    assert change["page"] == 1
    assert change["left_label"] == RULE
    assert change["right_label"] == SKY
    assert change["other_label"] == SEAL
    assert change["previous_version"] == 1
    # Every row cuts just right of the seal column, where the original
    # contrast is highest: 3 * 240**2 per row, twelve rows.
    assert change["cuts"] == [38] * 12
    assert change["score"] == 12 * 3 * 240**2
    # The search is a DP, not an enumeration of the 13**12 cut vectors
    # (cuts 36/37 are invalid: the seal pixel is not editable).
    assert change["search"]["cut_vectors"] == 13**12
    assert change["search"]["dp_states"] < 100_000

    # The revised label map is published with the new version.
    labels_response = client.get(
        f"/api/packages/{package_id}/versions/2/labels/1"
    )
    assert labels_response.status_code == 200
    revised = Image.open(io.BytesIO(labels_response.content))
    assert revised.mode == "L"
    pixels = revised.load()
    assert pixels[35, 20] == RULE  # flipped from SKY
    assert pixels[36, 20] == SEAL  # untouched
    assert pixels[37, 20] == RULE  # flipped from SKY
    assert pixels[38, 20] == SKY  # right of the cut
    assert pixels[24, 20] == RULE  # left of the cut, unchanged
    assert pixels[44, 20] == SKY  # outside the rect, unchanged
    assert pixels[10, 10] == PAPER  # far away, unchanged

    # The download bundle carries the revised label map and recoloured page.
    archive = zipfile.ZipFile(
        io.BytesIO(
            client.get(f"/api/packages/{package_id}/versions/2/download").content
        )
    )
    assert "labels_01.png" in archive.namelist()
    zipped = Image.open(io.BytesIO(archive.read("labels_01.png")))
    assert zipped.load()[37, 20] == RULE
    page = Image.open(io.BytesIO(archive.read("page_01.png")))
    palette_colours = {tuple(colour) for colour in PALETTE5_RGB}
    raw = page.tobytes()
    rendered = {(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)}
    assert rendered <= palette_colours

    # The superseded version now conflicts.
    stale = client.get(f"/api/packages/{package_id}/versions/1/download")
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "version_conflict"


def test_lock_excludes_highest_scoring_boundary(client: TestClient) -> None:
    created = create_demo_package(client)
    package_id = created["package_id"]

    # Locking the seal to red makes any rule/seal contact impossible: the
    # sky already touches the seal (so the sky must stay paper-compatible)
    # and the rule always touches the sky after the edit.
    locked = client.post(
        f"/api/packages/{package_id}/locks",
        json={"label": SEAL, "colour": "#a12f26"},
    )
    assert locked.status_code == 200, locked.text
    assert locked.json()["version"] == 2

    response = client.post(
        f"/api/packages/{package_id}/boundary-edits", json=edit_body(2)
    )
    assert response.status_code == 200, response.text
    change = response.json()["change"]
    # The highest-scoring boundary (cuts at 38/39, touching the seal) is
    # excluded by the lock; the best feasible one retreats left of the seal.
    assert change["cuts"] == [35] * 12
    assert change["score"] == 12 * (27**2 + 73**2 + 118**2)
    assert change["score"] < 12 * 3 * 240**2
    assert response.json()["locks"] == {"160": "#a12f26"}


def test_boundary_edit_stale_version_conflicts(client: TestClient) -> None:
    created = create_demo_package(client)
    package_id = created["package_id"]
    locked = client.post(
        f"/api/packages/{package_id}/locks",
        json={"label": SEAL, "colour": "#a12f26"},
    )
    assert locked.status_code == 200

    stale = client.post(
        f"/api/packages/{package_id}/boundary-edits", json=edit_body(1)
    )
    assert stale.status_code == 409, stale.text
    error = stale.json()["error"]
    assert error["code"] == "version_conflict"
    assert error["current_version"] == 2
    assert error["kept_version"] == 2

    # The package is untouched: still version 2 with the original label map.
    assert client.get(f"/api/packages/{package_id}").json()["version"] == 2
    labels = Image.open(
        io.BytesIO(
            client.get(
                f"/api/packages/{package_id}/versions/2/labels/1"
            ).content
        )
    )
    assert labels.load()[37, 20] == SKY


def test_boundary_edit_without_feasible_boundary(client: TestClient) -> None:
    # Every allowed cut puts a rule pixel next to the seal column, and the
    # seal already touches the sky outside the rect: with a two-colour
    # palette the resulting triangle cannot be coloured.
    width, height = 5, 7
    label_rows = [[SEAL, SKY, SKY, RULE, RULE] for _ in range(height)]
    rgb_rows = [[(0, 0, 0)] * width for _ in range(height)]
    rgb_files, label_files = upload_page(rgb_rows, label_rows)
    created = submit(
        client, rgb_files, label_files, ["#000000", "#ffffff"], 200
    )
    assert created.status_code == 201, created.text
    package_id = created.json()["package_id"]

    response = client.post(
        f"/api/packages/{package_id}/boundary-edits",
        json={
            "version": 1,
            "page": 1,
            "left_label": RULE,
            "right_label": SKY,
            "rect": {"x": 0, "y": 1, "width": 4, "height": 5},
        },
    )
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == "no_feasible_boundary"
    assert error["kept_version"] == 1
    assert error["details"]["other_label"] == SEAL
    # The current package was kept.
    assert client.get(f"/api/packages/{package_id}").json()["version"] == 1


def test_boundary_edit_tiebreak_and_row_smoothness(client: TestClient) -> None:
    # Row 0 scores only at cut 2, row 1 only at cut 5; the one-pixel
    # smoothness rule forbids jumping between them, and among the maximal
    # vectors the lexicographically left-most one wins.
    label_rows = [[1, 2, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 2, 2]]
    rgb_rows = [
        [(0, 0, 0), (0, 0, 0)] + [(255, 255, 255)] * 5,
        [(0, 0, 0)] * 5 + [(255, 255, 255)] * 2,
    ]
    rgb_files, label_files = upload_page(rgb_rows, label_rows)
    created = submit(
        client, rgb_files, label_files, ["#000000", "#ffffff"], 200
    )
    assert created.status_code == 201, created.text
    package_id = created.json()["package_id"]

    response = client.post(
        f"/api/packages/{package_id}/boundary-edits",
        json={
            "version": 1,
            "page": 1,
            "left_label": 1,
            "right_label": 2,
            "rect": {"x": 0, "y": 0, "width": 7, "height": 2},
        },
    )
    assert response.status_code == 200, response.text
    change = response.json()["change"]
    assert change["cuts"] == [2, 1]
    assert change["score"] == 3 * 255**2


def test_boundary_edit_validation(client: TestClient) -> None:
    created = create_demo_package(client)
    package_id = created["package_id"]

    def edit(**overrides):
        body = edit_body(1)
        body.update(overrides)
        return client.post(f"/api/packages/{package_id}/boundary-edits", json=body)

    # Rectangle wider than 24 pixels (schema validation).
    too_wide = edit_body(1)
    too_wide["rect"] = {"x": 0, "y": 0, "width": 25, "height": 4}
    response = client.post(
        f"/api/packages/{package_id}/boundary-edits", json=too_wide
    )
    assert response.status_code == 422

    # Rectangle outside the page.
    assert edit(rect={"x": 60, "y": 20, "width": 10, "height": 4}).status_code == 422
    # Same label on both sides.
    assert edit(right_label=RULE).status_code == 422
    # Unknown label.
    assert edit(left_label=7).status_code == 422
    # Page out of range.
    assert edit(page=2).status_code == 422
    # Rectangle plus its edge neighbours contain two other labels
    # (seal 160 and paper 240).
    assert (
        edit(rect={"x": 21, "y": 20, "width": 24, "height": 12}).status_code
        == 422
    )
    # A row without an adjacent pair of editable pixels.
    response = edit(rect={"x": 50, "y": 50, "width": 4, "height": 4})
    assert response.status_code == 422
    assert "no adjacent pair" in response.json()["error"]["message"]

    # Unknown package.
    missing = client.post("/api/packages/nope/boundary-edits", json=edit_body(1))
    assert missing.status_code == 404

    # Nothing was published by the failed attempts.
    assert client.get(f"/api/packages/{package_id}").json()["version"] == 1


def test_commit_race_keeps_current_package() -> None:
    # Solve a boundary edit against version 1, move the package on with a
    # lock, then try to publish: the edit must lose the race.
    rgb_rows, label_rows = boundary_demo_rows()
    page = make_page(1, 64, 64, label_rows, rgb_rows)
    local_store = PackageStore()
    analysis = analyse_pages([page], PALETTE5_RGB)
    created = local_store.create([page], PALETTE5_RGB, 200, analysis, {})
    package_id = created["package_id"]

    job = local_store.begin_boundary_edit(
        package_id,
        version=1,
        page_number=1,
        left=RULE,
        right=SKY,
        rect=Rect(x=24, y=20, width=16, height=12),
    )
    solution = solve_boundary(
        job.pages, job.analysis, job.palette, job.min_distance, job.locks, job.plan
    )
    assert solution is not None
    assert solution.cuts == [38] * 12

    # A concurrent lock bumps the version while the edit is being solved.
    moved = local_store.set_lock(package_id, SEAL, PALETTE5_RGB.index((161, 47, 38)))
    assert moved["version"] == 2

    with pytest.raises(VersionConflictError):
        local_store.commit_boundary_edit(package_id, job, solution)

    # The current package is kept: version 2, original label map.
    described = local_store.describe(package_id)
    assert described["version"] == 2
    assert described["locks"] == {"160": "#a12f26"}
    assert local_store.get(package_id).pages[0].labels[20 * 64 + 37] == SKY


def test_dp_matches_brute_force() -> None:
    """The DP must return exactly the brute-force best feasible boundary."""
    rng = random.Random(20261123)
    trials = 0
    attempts = 0
    while trials < 25 and attempts < 400:
        attempts += 1
        width, height = 9, 6
        split = rng.randint(2, 6)
        label_rows = [
            [1 if x <= split else 2 for x in range(width)] for _ in range(height)
        ]
        for y in range(height):
            for x in range(width):
                if rng.random() < 0.12:
                    label_rows[y][x] = 3
        rgb_rows = [
            [
                (rng.randrange(256), rng.randrange(256), rng.randrange(256))
                for _ in range(width)
            ]
            for _ in range(height)
        ]
        page = make_page(1, width, height, label_rows, rgb_rows)
        palette = list(
            dict.fromkeys(
                (
                    rng.randrange(256),
                    rng.randrange(256),
                    rng.randrange(256),
                )
                for _ in range(rng.randint(2, 4))
            )
        )
        if len(palette) < 2:
            continue
        min_distance = rng.choice([1, 100, 200, 300])
        rect = Rect(
            x=rng.randint(0, 3),
            y=rng.randint(0, 2),
            width=rng.randint(2, 4),
            height=rng.randint(1, 4),
        )
        if rect.x + rect.width > width or rect.y + rect.height > height:
            continue
        try:
            plan = plan_boundary_edit(page, 0, 1, 2, rect)
        except InputError:
            continue
        analysis = analyse_pages([page], palette)
        locks: dict[int, int] = {}
        if rng.random() < 0.4:
            locks = {rng.choice(analysis.labels): rng.randrange(len(palette))}

        solution = solve_boundary(
            [page], analysis, palette, min_distance, locks, plan
        )
        expected = _brute_force_boundary(page, plan, palette, min_distance, locks)
        if expected is None:
            assert solution is None, f"trial {trials}: DP found {solution}"
        else:
            assert solution is not None, f"trial {trials}: DP found nothing"
            assert (solution.score, solution.cuts) == expected, (
                f"trial {trials}: DP {(solution.score, solution.cuts)} != "
                f"brute force {expected}"
            )
        trials += 1
    assert trials == 25


def _brute_force_boundary(page, plan, palette, min_distance, locks):
    """Enumerate every smooth cut vector and fully re-solve each recolouring."""
    best = None
    rows = plan.rows

    def visit(i, previous_cut, score, cuts):
        nonlocal best
        if i == len(rows):
            new_labels = apply_cuts(page, plan, cuts)
            new_page = replace(page, labels=new_labels)
            analysis = analyse_pages([new_page], palette)
            index = {label: i for i, label in enumerate(analysis.labels)}
            result = solve_colouring(
                [analysis.costs[label] for label in analysis.labels],
                [(index[a], index[b]) for a, b in analysis.edges],
                palette,
                min_distance,
                {index[label]: c for label, c in locks.items()},
            )
            if result.assignment is None:
                return
            candidate = (score, list(cuts))
            if (
                best is None
                or candidate[0] > best[0]
                or (candidate[0] == best[0] and candidate[1] < best[1])
            ):
                best = candidate
            return
        row = rows[i]
        for cut in row.cuts:
            if previous_cut is not None and abs(cut - previous_cut) > 1:
                continue
            visit(i + 1, cut, score + row.scores[cut], cuts + [cut])

    visit(0, None, 0, [])
    return best
