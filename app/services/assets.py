"""Images in, and the URLs fal.ai needs to see them.

Two costs this module exists to avoid.

**Re-uploading.** An architect attaches the same massing view to six renders
across an afternoon. Uploading it six times is six round trips for six identical
files. Images are content-addressed by SHA-256, and a `fal_url` obtained for
those bytes once is reused everywhere, across projects and across users.

**Round-tripping a render that fal.ai is already hosting.** Continuing from a
finished render should not mean downloading it and pushing it back. The job row
already carries `output_url`; `from_job_output()` turns that into a reference
with no transfer at all.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app import db, fal_gateway
from app.config import Settings
from app.models import AssetKind

log = logging.getLogger(__name__)

# Formats the fal.ai endpoints in routing.yaml accept as input.
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

# What each suffix must actually decode as. PIL reports the real format, so a
# `.png` holding JPEG bytes is caught here rather than being stored under a name
# that lies about its contents — the suffix is what the file is written with and
# what every later reader trusts.
SUFFIX_FORMATS: dict[str, set[str]] = {
    ".png": {"png"},
    ".jpg": {"jpeg"},
    ".jpeg": {"jpeg"},
    ".webp": {"webp"},
}

# CAD files never reach fal.ai directly — they are converted to a PNG first —
# so they get their own suffix set and a larger cap. Architectural DXF and IFC
# exports with dense geometry routinely exceed the image limit.
CAD_ALLOWED_SUFFIXES = {".dxf", ".skp", ".ifc"}
MAX_CAD_UPLOAD_BYTES = 100 * 1024 * 1024


class AssetError(Exception):
    """Something about the supplied image makes it unusable. Nothing was spent."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_cad_signature(suffix: str, data: bytes) -> None:
    """Check that a CAD upload is what its extension claims, from the bytes.

    Extension alone decided this before, and the file was then handed straight
    to `ezdxf`, `ifcopenshell` or `openskp` — parsers with large C surfaces that
    were never meant to be a first line of defence against hostile input. This
    does not make them safe; it means they are only asked to parse files that at
    least begin like the format they handle.

    Deliberately a header check and nothing more. Validating the whole file is
    the parser's job, and duplicating it here would be a second implementation
    to keep in step with the first.
    """
    head = data[:4096]

    if suffix == ".dxf":
        # Binary DXF announces itself; ASCII DXF is group-code text that opens
        # a SECTION before anything else can appear.
        if head.startswith(b"AutoCAD Binary DXF"):
            return
        text = head.decode("latin-1", errors="ignore").upper()
        if "SECTION" in text and any(
            block in text for block in ("HEADER", "TABLES", "ENTITIES", "BLOCKS")
        ):
            return
        raise AssetError(
            "That file is named .dxf but does not contain a DXF header. Export "
            "it again as DXF (AutoCAD R12 or later)."
        )

    if suffix == ".ifc":
        # IFC-SPF is a STEP physical file; ifcXML is XML. Both are text.
        text = head.decode("latin-1", errors="ignore").lstrip("\N{ZERO WIDTH NO-BREAK SPACE} \t\r\n")
        if text.startswith("ISO-10303-21") or "ISO-10303-21" in text[:512]:
            return
        if text.lstrip().startswith("<?xml"):
            return
        raise AssetError(
            "That file is named .ifc but is not an IFC file — an IFC begins "
            "with ISO-10303-21. Export it again from your BIM tool."
        )

    if suffix == ".skp":
        # SketchUp 2013+ opens with a UTF-16LE "SketchUp Model" banner; older
        # versions are OLE2 compound documents.
        if b"S\x00k\x00e\x00t\x00c\x00h\x00U\x00p\x00" in head[:128]:
            return
        if b"SketchUp" in head[:128]:
            return
        if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            return
        raise AssetError(
            "That file is named .skp but does not look like a SketchUp model. "
            "Save it again from SketchUp."
        )


def _probe(data: bytes) -> tuple[int, int, str]:
    """Dimensions and real format, read from the bytes rather than the filename.

    A file called `plan.png` that is actually a PDF fails here, locally, in
    milliseconds — instead of at fal.ai after an upload, as an error about a
    field the architect never touched.
    """
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        with Image.open(BytesIO(data)) as img:
            return img.width, img.height, (img.format or "").lower()
    except (UnidentifiedImageError, OSError) as exc:
        raise AssetError(
            "That file is not an image the renderer can read. PNG, JPEG and "
            "WebP are supported."
        ) from exc


def _reload_asset(conn: sqlite3.Connection, asset_id: int) -> sqlite3.Row:
    """Read an asset back after inserting it.

    This was `assert row is not None`, which `python -O` removes entirely — and
    then every caller dereferences None. It can only fail if the row was deleted
    between the two statements, so it stays an exception the caller can report.
    """
    row = db.get_asset(conn, asset_id)
    if row is None:
        raise AssetError(
            f"Asset {asset_id} vanished immediately after it was stored."
        )
    return row


def _storage_path(settings: Settings, sha: str, suffix: str) -> Path:
    """Content-addressed on disk: identical bytes occupy one file, once.

    Sharded by the first two hex characters so no single directory accumulates
    every image the office has ever uploaded.
    """
    return settings.inputs_dir() / sha[:2] / f"{sha}{suffix}"


def ingest(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    data: bytes,
    filename: str,
    project_id: int | None = None,
    uploaded_by: int | None = None,
    kind: AssetKind = AssetKind.UPLOAD,
) -> sqlite3.Row:
    """Store an uploaded image and return its asset row.

    Re-uploading the same bytes into the same project returns the existing
    asset rather than creating a second one, so a gallery does not fill with
    duplicates and the cached fal.ai URL keeps being used.
    """
    if not data:
        raise AssetError("That file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise AssetError(
            f"That file is {len(data) / 1_048_576:.0f} MB. The limit is "
            f"{MAX_UPLOAD_BYTES // 1_048_576} MB — export a smaller version."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise AssetError(
            f"{suffix or 'That file type'} is not supported. Use PNG, JPEG or WebP."
        )

    width, height, fmt = _probe(data)
    expected = SUFFIX_FORMATS[suffix]
    if fmt not in expected:
        # The bytes decode fine, but as something else. Storing them under a
        # name that lies is how a `.png` that is really a JPEG reaches fal.ai
        # and comes back as an error about a field nobody touched.
        raise AssetError(
            f"That file is named {suffix} but the image inside it is "
            f"{fmt.upper() or 'an unknown format'}. Save it as "
            f"{min(expected).upper()}, or rename it to match."
        )
    sha = sha256_bytes(data)

    existing = db.find_asset_by_sha(conn, sha, project_id=project_id)
    if existing is not None:
        return existing

    dest = _storage_path(settings, sha, suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)

    # These exact bytes may already be hosted from another project or another
    # user. Carrying the URL over is the difference between an instant attach
    # and a fresh upload.
    cached_url = _hosted(db.find_fal_url_by_sha(conn, sha), settings)

    try:
        asset_id = db.create_asset(
            conn,
            sha256=sha,
            project_id=project_id,
            uploaded_by=uploaded_by,
            local_path=str(dest),
            fal_url=cached_url,
            width=width,
            height=height,
            kind=kind,
        )
    except db.ForeignKeyViolation as exc:
        # A project_id that does not exist. There was no handling here at all,
        # so it escaped as an IntegrityError and arrived as a 500.
        raise AssetError(str(exc)) from exc
    return _reload_asset(conn, asset_id)


def ingest_cad(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    data: bytes,
    filename: str,
    project_id: int | None = None,
    uploaded_by: int | None = None,
) -> sqlite3.Row:
    """Store an uploaded DXF/SKP file and return its asset row.

    Mirrors `ingest()`'s content-addressing — re-attaching the same file for a
    second conversion attempt returns the existing row rather than storing a
    duplicate — but there is no image to probe and no fal.ai URL to cache: a
    CAD file is never itself sent to fal.ai, only the PNG a conversion of it
    produces is, and that PNG is ingested separately through `ingest()`.
    """
    if not data:
        raise AssetError("That file is empty.")
    if len(data) > MAX_CAD_UPLOAD_BYTES:
        raise AssetError(
            f"That file is {len(data) / 1_048_576:.0f} MB. The limit is "
            f"{MAX_CAD_UPLOAD_BYTES // 1_048_576} MB."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in CAD_ALLOWED_SUFFIXES:
        raise AssetError(
            f"{suffix or 'That file type'} is not a supported CAD format. "
            "Use DXF, IFC or SKP."
        )
    _verify_cad_signature(suffix, data)

    sha = sha256_bytes(data)
    existing = db.find_asset_by_sha(conn, sha, project_id=project_id)
    if existing is not None:
        return existing

    dest = _storage_path(settings, sha, suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)

    try:
        asset_id = db.create_asset(
            conn,
            sha256=sha,
            project_id=project_id,
            uploaded_by=uploaded_by,
            local_path=str(dest),
            fal_url=None,
            width=None,
            height=None,
            kind=AssetKind.CAD_UPLOAD,
        )
    except db.ForeignKeyViolation as exc:
        raise AssetError(str(exc)) from exc
    return _reload_asset(conn, asset_id)


def _hosted(url: str | None, settings: Settings) -> str | None:
    """A stored URL fal.ai can fetch, or None.

    A `fake://` URL names a file only the fake engine can read. Under the fal
    engine it counts as absent, so the image is uploaded afresh instead of being
    sent to fal.ai as a reference.
    """
    if not url:
        return None
    if settings.render_engine == "fal" and url.startswith("fake://"):
        return None
    return url


def ensure_fal_url(conn: sqlite3.Connection, settings: Settings, asset_id: int) -> str:
    """The URL fal.ai can fetch this image from, uploading only if necessary."""
    row = db.get_asset(conn, asset_id)
    if row is None:
        raise AssetError(f"No asset {asset_id}.")

    if stored := _hosted(row["fal_url"], settings):
        return stored

    cached = _hosted(db.find_fal_url_by_sha(conn, str(row["sha256"])), settings)
    if cached:
        db.set_asset_fal_url(conn, asset_id, cached)
        return cached

    local = row["local_path"]
    if not local or not Path(str(local)).exists():
        raise AssetError(
            f"Asset {asset_id} has no local copy and no uploaded URL — there is "
            "nothing left to send. Re-attach the image."
        )

    url = fal_gateway.upload_input(settings, Path(str(local)))
    db.set_asset_fal_url(conn, asset_id, url)
    return url


def from_job_output(
    conn: sqlite3.Connection,
    settings: Settings,
    job_id: int,
    *,
    project_id: int | None = None,
    uploaded_by: int | None = None,
    output_index: int = 0,
) -> sqlite3.Row:
    """Turn one image of a finished render into a reusable reference image.

    This is what "edit this" and "use this as the style anchor" both stand on.
    `output_index` picks which of the render's images; 0 is the first.

    The fast path costs nothing: the job already recorded the URL fal.ai serves
    the render from, so the asset is created pointing straight at it. If that URL
    has aged out — fal's retention is documented ambiguously and has never been
    observed expiring here — the locally saved copy is ingested instead, which
    re-uploads once and then behaves identically.
    """
    job = db.get_job(conn, job_id)
    if job is None:
        raise AssetError(f"No job {job_id}.")
    if job["status"] != "completed":
        raise AssetError(
            f"Job {job_id} is {job['status']}. Only a finished render can be "
            "used as a reference."
        )

    # Which image of the job. A job that predates job_outputs has no rows, and
    # its only image is index 0 on the job row itself.
    output = db.get_job_output(conn, job_id, output_index)
    if output is None and output_index != 0:
        raise AssetError(f"Job {job_id} has no image {output_index}.")
    if output is not None:
        local_path = output["path"]
        output_url = output["url"]
    else:
        local_path = job["output_path"]
        output_url = job["output_url"]

    sha: str | None = None
    if local_path and Path(str(local_path)).exists():
        data = Path(str(local_path)).read_bytes()
        sha = sha256_bytes(data)
        existing = db.find_asset_by_sha(conn, sha, project_id=project_id)
        if existing is not None:
            return existing

    output_url = _hosted(output_url, settings)
    if output_url:
        width = height = None
        if local_path and Path(str(local_path)).exists():
            try:
                width, height, _ = _probe(Path(str(local_path)).read_bytes())
            except AssetError:
                pass
        try:
            asset_id = db.create_asset(
                conn,
                sha256=sha or f"job:{job_id}",
                project_id=project_id if project_id is not None else job["project_id"],
                uploaded_by=uploaded_by,
                local_path=str(local_path) if local_path else None,
                fal_url=str(output_url),
                width=width,
                height=height,
                kind=AssetKind.RENDER_OUTPUT,
            )
        except db.ForeignKeyViolation as exc:
            raise AssetError(str(exc)) from exc
        return _reload_asset(conn, asset_id)

    # No hosted URL — fall back to the local copy, which costs one upload.
    if not local_path or not Path(str(local_path)).exists():
        raise AssetError(
            f"Job {job_id} has no image left to continue from — neither a saved "
            "file nor a hosted URL."
        )
    return ingest(
        conn, settings,
        data=Path(str(local_path)).read_bytes(),
        filename=Path(str(local_path)).name,
        project_id=project_id if project_id is not None else job["project_id"],
        uploaded_by=uploaded_by,
        kind=AssetKind.RENDER_OUTPUT,
    )


def ingest_mask(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    data: bytes,
    source_asset_id: int,
    project_id: int | None = None,
    uploaded_by: int | None = None,
) -> sqlite3.Row:
    """Store a mask, refusing one that does not match its source image.

    `flux-pro/v1/fill` requires the mask to match the source's dimensions
    exactly and fails otherwise. Checking here means the mismatch is reported
    while the architect is still looking at the mask they drew, rather than as a
    422 several steps later.
    """
    source = db.get_asset(conn, source_asset_id)
    if source is None:
        raise AssetError(f"No asset {source_asset_id} to mask.")

    width, height, _ = _probe(data)
    known = source["width"] and source["height"]
    if known and (width, height) != (int(source["width"]), int(source["height"])):
        raise AssetError(
            f"The mask is {width}×{height} but the image is "
            f"{source['width']}×{source['height']}. This endpoint requires "
            "them to match exactly."
        )

    return ingest(
        conn, settings,
        data=data,
        filename="mask.png",
        project_id=project_id,
        uploaded_by=uploaded_by,
        kind=AssetKind.MASK,
    )
