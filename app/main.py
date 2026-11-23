"""FastAPI application for the e-paper colouring service."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .boundary import Rect, solve_boundary
from .imaging import (
    MAX_PAGES,
    MAX_UPLOAD_BYTES,
    InputError,
    load_page,
    parse_lock_colour,
    parse_min_distance,
    parse_palette,
)
from .pipeline import analyse_pages
from .store import (
    InfeasibleError,
    NotFoundError,
    PackageStore,
    VersionConflictError,
    VersionNotFoundError,
)

MAX_LOCK_ENTRIES = 12

app = FastAPI(
    title="E-paper colouring service",
    version="1.0.0",
    description=(
        "Assigns a fixed e-paper palette to labelled page maps and serves "
        "the rendered PNG pages plus a legend. Colours are solved exactly "
        "(minimum summed squared RGB error) by branch and bound, not by "
        "enumerating all colourings."
    ),
)

store = PackageStore()


# --------------------------------------------------------------------------
# request models
# --------------------------------------------------------------------------
class LockRequest(BaseModel):
    label: int = Field(..., ge=0, le=255, description="grayscale label value")
    colour: Any = Field(
        ...,
        description="locked colour, either #rrggbb or [r, g, b], must exist in palette",
    )


class RectModel(BaseModel):
    x: int = Field(..., ge=0, description="left edge of the edit rectangle")
    y: int = Field(..., ge=0, description="top edge of the edit rectangle")
    width: int = Field(..., ge=1, le=24, description="at most 24 pixels")
    height: int = Field(..., ge=1, le=24, description="at most 24 pixels")


class BoundaryEditRequest(BaseModel):
    version: int = Field(
        ..., ge=1, description="current package version the edit is based on"
    )
    page: int = Field(..., ge=1, description="1-based page number to edit")
    left_label: int = Field(
        ..., ge=0, le=255, description="label assigned left of each cut"
    )
    right_label: int = Field(
        ..., ge=0, le=255, description="label assigned right of each cut"
    )
    rect: RectModel


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    if extra:
        payload["error"].update(extra)
    return JSONResponse(status_code=status, content=payload)


def _version_conflict(package_id: str, exc: VersionConflictError) -> JSONResponse:
    """409 for a stale base version, reporting the kept current package."""
    try:
        kept = store.describe(package_id)
    except NotFoundError:
        return _error(404, "package_not_found", "no such package")
    return _error(
        409,
        "version_conflict",
        str(exc),
        current_version=exc.current_version,
        kept_version=kept["version"],
        kept_locks=kept["locks"],
        kept_download_url=kept["download_url"],
    )


def _parse_locks(raw: str | None, palette) -> dict[int, int]:
    if raw is None or raw.strip() == "":
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise InputError("locks must be a JSON object of label -> colour")
    if not isinstance(data, dict):
        raise InputError("locks must be a JSON object of label -> colour")
    if len(data) > MAX_LOCK_ENTRIES:
        raise InputError(f"at most {MAX_LOCK_ENTRIES} locks may be submitted")
    result: dict[int, int] = {}
    index = {colour: i for i, colour in enumerate(palette)}
    for key, value in data.items():
        try:
            label = int(key)
        except (TypeError, ValueError):
            raise InputError(f"lock label {key!r} is not an integer")
        if not 0 <= label <= 255:
            raise InputError(f"lock label {label} is outside 0..255")
        colour = parse_lock_colour(value)
        if colour not in index:
            raise InputError(
                f"lock colour {colour} for label {label} is not in the palette"
            )
        result[label] = index[colour]
    return result


async def _read_upload(upload: UploadFile, field: str) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise InputError(f"{field}: file exceeds {MAX_UPLOAD_BYTES} bytes")
    if not data:
        raise InputError(f"{field}: file is empty")
    return data


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/packages", status_code=201)
async def create_package(
    rgb_pages: list[UploadFile] = File(  # noqa: B008 - FastAPI dependency style
        ..., description="1-4 RGB PNG originals"
    ),
    label_pages: list[UploadFile] = File(  # noqa: B008 - FastAPI dependency style
        ..., description="matching 8-bit grayscale PNGs"
    ),
    palette: str = Form(..., description='JSON array, e.g. ["#000000", "#ffffff"]'),
    min_distance: float = Form(
        ..., description="minimum RGB Euclidean colour distance"
    ),
    locks: str | None = Form(
        None, description='optional JSON object {"12": "#ff0000"}'
    ),
):
    try:
        colours = parse_palette(palette)
        distance = parse_min_distance(min_distance)
        page_count = len(rgb_pages)
        if not 1 <= page_count <= MAX_PAGES:
            raise InputError(
                f"submit between 1 and {MAX_PAGES} pages, got {page_count}"
            )
        if len(label_pages) != page_count:
            raise InputError(
                f"expected {page_count} label maps to match {page_count} RGB pages, "
                f"got {len(label_pages)}"
            )
        pages = []
        for index, (rgb_file, label_file) in enumerate(
            zip(rgb_pages, label_pages), start=1
        ):
            rgb_name = rgb_file.filename or f"page {index} rgb"
            label_name = label_file.filename or f"page {index} labels"
            pages.append(
                load_page(
                    await _read_upload(rgb_file, rgb_name),
                    await _read_upload(label_file, label_name),
                    index,
                    rgb_name,
                    label_name,
                )
            )
        analysis = analyse_pages(pages, colours)
        if not analysis.labels:
            raise InputError("the label maps contain no pixels")
        requested_locks = _parse_locks(locks, colours)
        unknown = sorted(set(requested_locks) - set(analysis.labels))
        if unknown:
            raise InputError(f"locked labels not present in the pages: {unknown}")
        # Solving is CPU-bound; keep it off the event loop.
        snapshot = await run_in_threadpool(
            store.create, pages, colours, distance, analysis, requested_locks
        )
    except InputError as exc:
        return _error(422, "invalid_input", str(exc))
    except InfeasibleError as exc:
        return _error(
            409,
            "infeasible",
            exc.details.get("message", "no feasible colouring"),
            details=exc.details,
        )
    return snapshot


@app.get("/api/packages/{package_id}")
def get_package(package_id: str):
    try:
        return store.describe(package_id)
    except NotFoundError:
        return _error(404, "package_not_found", "no such package")


@app.post("/api/packages/{package_id}/locks")
async def lock_label(package_id: str, request: LockRequest):
    try:
        package = store.get(package_id)
        colour = parse_lock_colour(request.colour)
        try:
            colour_index = package.palette.index(colour)
        except ValueError:
            return _error(
                422,
                "invalid_input",
                f"lock colour {colour} is not one of the package palette colours",
            )
        return await run_in_threadpool(
            store.set_lock, package_id, request.label, colour_index
        )
    except NotFoundError:
        return _error(404, "package_not_found", "no such package")
    except KeyError as exc:
        return _error(422, "invalid_input", str(exc.args[0]))
    except InfeasibleError as exc:
        kept = store.describe(package_id)
        return _error(
            409,
            "infeasible",
            exc.details.get("message", "no feasible colouring"),
            details=exc.details,
            kept_version=kept["version"],
            kept_locks=kept["locks"],
            kept_download_url=kept["download_url"],
        )


@app.delete("/api/packages/{package_id}/locks/{label}")
async def unlock_label(package_id: str, label: int):
    try:
        return await run_in_threadpool(store.clear_lock, package_id, label)
    except NotFoundError:
        return _error(404, "package_not_found", "no such package")
    except KeyError as exc:
        return _error(422, "invalid_input", str(exc.args[0]))
    except InfeasibleError as exc:
        kept = store.describe(package_id)
        return _error(
            409,
            "infeasible",
            exc.details.get("message", "no feasible colouring"),
            details=exc.details,
            kept_version=kept["version"],
            kept_locks=kept["locks"],
            kept_download_url=kept["download_url"],
        )


@app.post("/api/packages/{package_id}/boundary-edits")
async def boundary_edit(package_id: str, request: BoundaryEditRequest):
    """Re-draw the boundary between two labels inside a small rectangle.

    The edit is validated and snapshotted under the package lock, solved
    off-lock (other edits may proceed), and published only if the package
    is still at the requested version.
    """
    rect = Rect(
        x=request.rect.x,
        y=request.rect.y,
        width=request.rect.width,
        height=request.rect.height,
    )
    try:
        job = store.begin_boundary_edit(
            package_id,
            version=request.version,
            page_number=request.page,
            left=request.left_label,
            right=request.right_label,
            rect=rect,
        )
    except NotFoundError:
        return _error(404, "package_not_found", "no such package")
    except InputError as exc:
        return _error(422, "invalid_input", str(exc))
    except VersionConflictError as exc:
        return _version_conflict(package_id, exc)

    # The search is CPU-bound and must not block other edits: run it on the
    # snapshot in a worker thread, without holding the package lock.
    solution = await run_in_threadpool(
        solve_boundary,
        job.pages,
        job.analysis,
        job.palette,
        job.min_distance,
        job.locks,
        job.plan,
    )
    if solution is None:
        kept = store.describe(package_id)
        return _error(
            409,
            "no_feasible_boundary",
            "no cut placement inside the rectangle keeps the whole set "
            "colourable under the minimum colour distance and the active "
            "locks; the current package was kept",
            details={
                "page": request.page,
                "left_label": request.left_label,
                "right_label": request.right_label,
                "other_label": job.plan.other,
                "rect": request.rect.model_dump(),
                "locks": kept["locks"],
            },
            kept_version=kept["version"],
            kept_locks=kept["locks"],
            kept_download_url=kept["download_url"],
        )
    try:
        return await run_in_threadpool(
            store.commit_boundary_edit, package_id, job, solution
        )
    except VersionConflictError as exc:
        return _version_conflict(package_id, exc)
    except InfeasibleError as exc:
        # Defensive: the pre-checked feasibility cannot change while the
        # version is pinned, but never leave the package half-edited.
        kept = store.describe(package_id)
        return _error(
            409,
            "infeasible",
            exc.details.get("message", "no feasible colouring"),
            details=exc.details,
            kept_version=kept["version"],
            kept_locks=kept["locks"],
            kept_download_url=kept["download_url"],
        )


@app.get("/api/packages/{package_id}/versions/{version}/download")
def download_version(package_id: str, version: int):
    try:
        bundle, data = store.download(package_id, version)
    except VersionNotFoundError as exc:
        return _error(404, "version_not_found", str(exc))
    except NotFoundError:
        return _error(404, "package_not_found", "no such package")
    except VersionConflictError as exc:
        return _error(
            409,
            "version_conflict",
            str(exc),
            current_version=exc.current_version,
            current_download_url=(
                f"/api/packages/{package_id}/versions/{exc.current_version}/download"
            ),
        )
    headers = {
        "Content-Disposition": (
            f'attachment; filename="epaper-{package_id[:8]}-v{bundle.number}.zip"'
        ),
        "X-Package-Version": str(bundle.number),
    }
    return Response(content=data, media_type="application/zip", headers=headers)


@app.get("/api/packages/{package_id}/versions/{version}/legend.json")
def download_legend(package_id: str, version: int):
    try:
        bundle = store.get_bundle(package_id, version)
    except VersionNotFoundError as exc:
        return _error(404, "version_not_found", str(exc))
    except NotFoundError:
        return _error(404, "package_not_found", "no such package")
    except VersionConflictError as exc:
        return _error(
            409,
            "version_conflict",
            str(exc),
            current_version=exc.current_version,
        )
    return Response(content=bundle.legend_json, media_type="application/json")


@app.get("/api/packages/{package_id}/versions/{version}/pages/{page}")
def download_page(package_id: str, version: int, page: int):
    try:
        bundle = store.get_bundle(package_id, version)
    except VersionNotFoundError as exc:
        return _error(404, "version_not_found", str(exc))
    except NotFoundError:
        return _error(404, "package_not_found", "no such package")
    except VersionConflictError as exc:
        return _error(
            409,
            "version_conflict",
            str(exc),
            current_version=exc.current_version,
        )
    if not 1 <= page <= len(bundle.pages_png):
        return _error(404, "page_not_found", f"page {page} does not exist")
    return Response(
        content=bundle.pages_png[page - 1],
        media_type="image/png",
        headers={"X-Package-Version": str(bundle.number)},
    )


@app.get("/api/packages/{package_id}/versions/{version}/labels/{page}")
def download_labels(package_id: str, version: int, page: int):
    """The label map published with this version (revised by boundary edits)."""
    try:
        bundle = store.get_bundle(package_id, version)
    except VersionNotFoundError as exc:
        return _error(404, "version_not_found", str(exc))
    except NotFoundError:
        return _error(404, "package_not_found", "no such package")
    except VersionConflictError as exc:
        return _error(
            409,
            "version_conflict",
            str(exc),
            current_version=exc.current_version,
        )
    if not 1 <= page <= len(bundle.labels_png):
        return _error(404, "page_not_found", f"page {page} does not exist")
    return Response(
        content=bundle.labels_png[page - 1],
        media_type="image/png",
        headers={"X-Package-Version": str(bundle.number)},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        problems.append(f"{location}: {error.get('msg', 'invalid value')}")
    return _error(
        422,
        "invalid_input",
        "request validation failed: " + "; ".join(problems),
    )


@app.exception_handler(InputError)
async def input_error_handler(request: Request, exc: InputError):
    return _error(422, "invalid_input", str(exc))


@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    return _error(404, "package_not_found", "no such package")
