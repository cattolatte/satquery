"""
Input inspection and compatibility checking.

The problem statement makes this a required controller step:

    "check the number, modality, format, metadata, and compatibility of the
    input images"

It is also the cheapest place to fail. A bi-temporal pair whose two images
cover different ground, or an optical-SAR pair that is not co-registered, will
produce a confident and wrong answer from any downstream model. Rejecting here
is far better than explaining later.
"""
from __future__ import annotations

from pathlib import Path

from ..schema import ImageMeta, InputKind, Modality

GEO_EXT = {".tif", ".tiff"}
BENCH_EXT = {".png", ".jpg", ".jpeg"}


def _band_modality(bands: int, name: str, desc: list[str], fmt: str = "") -> Modality:
    """Guess modality from band count, filename and band descriptions.

    SAR products are typically 1-2 bands (amplitude, or a polarimetric pair);
    optical/multispectral is 3+ (RGB) or many (Sentinel-2 has 12-13). The
    filename is checked first because it is usually right and always cheap.
    """
    n = name.lower()
    if any(k in n for k in ("sar", "risat", "sentinel-1", "s1_", "grd", "slc", "_vv", "_vh")):
        return Modality.SAR
    if any(k in n for k in ("optical", "cartosat", "sentinel-2", "s2_", "msi", "rgb", "toa")):
        return Modality.OPTICAL
    blob = " ".join(desc).lower()
    if any(k in blob for k in ("vv", "vh", "hh", "hv", "amplitude", "sigma0")):
        return Modality.SAR
    if bands >= 3:
        return Modality.OPTICAL

    # Band count implies SAR only for geospatial products. A grayscale PNG or
    # JPEG is an ordinary photograph -- VRSBench is full of them, and 540 of its
    # gold answers are literally "grayscale". Reading those as radar made two
    # plain images look like a co-registered optical-SAR pair and routed them
    # to the cross-modal tool.
    if fmt in ("GeoTIFF", "TIFF"):
        return Modality.SAR if bands <= 2 else Modality.UNKNOWN
    return Modality.OPTICAL


def inspect_image(path: str | Path) -> ImageMeta:
    """Read metadata without loading pixels where possible."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in GEO_EXT:
        return _inspect_geotiff(p)
    if ext in BENCH_EXT:
        return _inspect_plain(p)

    # An unfamiliar extension is not the same as an unreadable file. CDVQA
    # ships its frames as ".img" and they are ordinary PNGs; rejecting them on
    # the name meant the controller refused every bi-temporal pair in the
    # benchmark as unreadable, and would refuse an upload named the same way.
    # Sniff the content before giving up, and record what the name claimed.
    meta = _inspect_plain(p)
    if meta.width:
        meta.notes.append(f"extension {ext!r} not recognised; identified by content")
        return meta
    return ImageMeta(str(p), ext.lstrip(".") or "unknown", 0, 0, 0,
                     notes=[f"unsupported extension {ext!r}"])


def _inspect_geotiff(p: Path) -> ImageMeta:
    try:
        import rasterio
    except ImportError:
        return ImageMeta(str(p), "GeoTIFF", 0, 0, 0,
                         notes=["rasterio not installed; cannot read geospatial metadata"])
    try:
        with rasterio.open(p) as src:
            desc = [d or "" for d in (src.descriptions or [])]
            bounds = tuple(src.bounds) if src.crs else None
            meta = ImageMeta(
                path=str(p), fmt="GeoTIFF", width=src.width, height=src.height,
                bands=src.count, georeferenced=src.crs is not None,
                crs=str(src.crs) if src.crs else None, bounds=bounds,
                acquired=(src.tags().get("TIFFTAG_DATETIME")
                          or src.tags().get("ACQUISITION_DATE")),
            )
            meta.modality = _band_modality(src.count, p.name, desc, "GeoTIFF")
            if not meta.georeferenced:
                meta.notes.append("no CRS: cannot verify spatial correspondence")
            return meta
    except Exception as e:                                  # noqa: BLE001
        return ImageMeta(str(p), "GeoTIFF", 0, 0, 0, notes=[f"unreadable: {e}"])


def _inspect_plain(p: Path) -> ImageMeta:
    """PNG/JPEG are accepted ONLY for the prescribed public benchmarks, which
    the problem statement states explicitly. The restriction is recorded here
    rather than enforced, because the benchmark harness legitimately uses them.
    """
    try:
        from PIL import Image
        with Image.open(p) as im:
            w, h = im.size
            bands = len(im.getbands())
    except Exception as e:                                  # noqa: BLE001
        return ImageMeta(str(p), p.suffix.lstrip(".").upper(), 0, 0, 0,
                         notes=[f"unreadable: {e}"])
    return ImageMeta(
        path=str(p), fmt=p.suffix.lstrip(".").upper(), width=w, height=h, bands=bands,
        modality=_band_modality(bands, p.name, [], p.suffix.lstrip(".").upper()),
        notes=["non-geospatial format: permitted only for benchmark datasets"],
    )


def classify_inputs(metas: list[ImageMeta]) -> tuple[InputKind, list[str]]:
    """Decide which of the three input configurations this is, and why not the
    others. Returns the kind plus the reasons anything was rejected."""
    reasons: list[str] = []

    if not metas:
        return InputKind.INVALID, ["no images supplied"]
    for m in metas:
        if m.width == 0 or m.height == 0:
            reasons.append(f"{Path(m.path).name}: unreadable")
    if reasons:
        return InputKind.INVALID, reasons

    if len(metas) == 1:
        return InputKind.SINGLE, reasons
    if len(metas) > 2:
        return InputKind.INVALID, [f"{len(metas)} images supplied; scope is one or two"]

    a, b = metas
    same_modality = a.modality == b.modality and a.modality != Modality.UNKNOWN
    mixed = {a.modality, b.modality} == {Modality.OPTICAL, Modality.SAR}

    if mixed:
        kind = InputKind.CROSS_MODAL_PAIR
    elif same_modality:
        kind = InputKind.BI_TEMPORAL_PAIR
    else:
        return InputKind.INVALID, [
            f"cannot pair {a.modality.value} with {b.modality.value}: "
            "expected optical+SAR (cross-modal) or matching modalities (bi-temporal)"
        ]

    # Both pair types require spatial correspondence. Without georeferencing we
    # cannot verify it, which is a warning rather than a rejection: benchmark
    # datasets ship pre-paired PNGs with no CRS at all.
    if a.georeferenced and b.georeferenced:
        if a.crs != b.crs:
            reasons.append(f"CRS mismatch: {a.crs} vs {b.crs}")
        elif a.bounds and b.bounds and not _overlaps(a.bounds, b.bounds):
            reasons.append("image bounds do not overlap: not the same area")
    else:
        reasons.append("not georeferenced: spatial correspondence assumed, not verified")

    if kind is InputKind.BI_TEMPORAL_PAIR and a.acquired and b.acquired and a.acquired == b.acquired:
        reasons.append("identical acquisition dates: a bi-temporal pair should differ")

    # A hard failure downgrades to INVALID; a soft note is carried forward.
    hard = [r for r in reasons if "mismatch" in r or "do not overlap" in r]
    return (InputKind.INVALID if hard else kind), reasons


def _overlaps(a: tuple, b: tuple) -> bool:
    """Do two (left, bottom, right, top) boxes intersect at all?"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
