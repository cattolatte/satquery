"""Input inspection: format, modality and pair classification.

Modality decides which specialist runs, so getting it wrong routes a query to
a tool built for a different sensor. The bug this guards: band count alone
implied SAR, so a grayscale photograph was read as radar and two ordinary
images became a co-registered optical-SAR pair.
"""
from pathlib import Path

import pytest

from satquery.io.inspect import classify_inputs, inspect_image
from satquery.schema import InputKind, Modality

Image = pytest.importorskip("PIL.Image")


def _png(directory, name, mode="RGB", size=(64, 64)):
    path = Path(directory) / name
    Image.new(mode, size).save(path)
    return str(path)


class TestModality:
    def test_grayscale_photograph_is_optical_not_radar(self, tmp_path):
        """VRSBench is full of grayscale aerial photos -- 540 of its gold
        answers are the word "grayscale". Reading those as SAR is wrong."""
        assert inspect_image(_png(tmp_path, "gray.png", "L")).modality is Modality.OPTICAL

    def test_rgb_is_optical(self, tmp_path):
        assert inspect_image(_png(tmp_path, "rgb.png")).modality is Modality.OPTICAL

    def test_filename_still_identifies_sar(self, tmp_path):
        assert inspect_image(_png(tmp_path, "s1_grd_vv.png", "L")).modality is Modality.SAR

    def test_sentinel2_name_is_optical(self, tmp_path):
        assert inspect_image(_png(tmp_path, "s2_msi_scene.png", "L")).modality is Modality.OPTICAL


class TestPairClassification:
    def test_two_plain_photos_are_not_a_cross_modal_pair(self, tmp_path):
        metas = [inspect_image(_png(tmp_path, "a.png", "L")),
                 inspect_image(_png(tmp_path, "b.png"))]
        assert classify_inputs(metas)[0] is not InputKind.CROSS_MODAL_PAIR

    def test_optical_and_named_sar_are_cross_modal(self, tmp_path):
        metas = [inspect_image(_png(tmp_path, "rgb.png")),
                 inspect_image(_png(tmp_path, "s1_grd_vv.png", "L"))]
        assert classify_inputs(metas)[0] is InputKind.CROSS_MODAL_PAIR

    def test_single_image(self, tmp_path):
        assert classify_inputs([inspect_image(_png(tmp_path, "a.png"))])[0] is InputKind.SINGLE


class TestUnreadableInputs:
    """Bad input must be reported, never raised: the trace records the reason
    and the controller refuses, rather than the request failing with a stack."""

    def test_missing_file(self, tmp_path):
        meta = inspect_image(str(Path(tmp_path) / "nope.png"))
        assert meta.width == 0 and any("unreadable" in n for n in meta.notes)

    def test_empty_file(self, tmp_path):
        path = Path(tmp_path) / "empty.png"
        path.write_bytes(b"")
        meta = inspect_image(str(path))
        assert meta.bands == 0 and meta.notes

    def test_corrupt_geotiff(self, tmp_path):
        path = Path(tmp_path) / "broken.tif"
        path.write_bytes(b"not a tiff at all")
        meta = inspect_image(str(path))
        assert meta.bands == 0 and meta.notes

    def test_unreadable_inputs_yield_no_usable_task(self, tmp_path):
        meta = inspect_image(str(Path(tmp_path) / "gone.png"))
        kind, notes = classify_inputs([meta])
        assert kind is InputKind.INVALID or notes

    def test_non_geospatial_format_is_flagged(self, tmp_path):
        """The statement permits PNG/JPEG only for the prescribed benchmarks."""
        meta = inspect_image(_png(tmp_path, "a.png"))
        assert any("non-geospatial" in n for n in meta.notes)


class TestUnknownExtension:
    """An unfamiliar extension is not the same as an unreadable file.

    CDVQA ships its frames as ".img" and they are ordinary PNGs. Rejecting them
    on the name meant the controller refused every bi-temporal pair in the
    benchmark as unreadable, while the direct tool path -- which builds its own
    metadata -- worked fine. The benchmark score looked like a routing problem
    and was an input-handling one.
    """

    def test_png_content_under_an_unknown_extension_is_read(self, tmp_path):
        from PIL import Image
        path = Path(tmp_path) / "frame.img"
        Image.new("RGB", (64, 64)).save(path, format="PNG")
        meta = inspect_image(str(path))
        assert meta.width == 64 and meta.bands == 3

    def test_the_unrecognised_name_is_recorded(self, tmp_path):
        from PIL import Image
        path = Path(tmp_path) / "frame.img"
        Image.new("RGB", (32, 32)).save(path, format="PNG")
        assert any("not recognised" in n for n in inspect_image(str(path)).notes)

    def test_a_genuinely_unreadable_file_still_fails(self, tmp_path):
        path = Path(tmp_path) / "junk.img"
        path.write_bytes(b"not an image")
        meta = inspect_image(str(path))
        assert meta.width == 0 and meta.notes

    def test_such_a_pair_classifies_as_bi_temporal(self, tmp_path):
        from PIL import Image
        paths = []
        for i in range(2):
            p = Path(tmp_path) / f"f{i}.img"
            Image.new("RGB", (64, 64)).save(p, format="PNG")
            paths.append(str(p))
        kind, _ = classify_inputs([inspect_image(p) for p in paths])
        assert kind is InputKind.BI_TEMPORAL_PAIR
