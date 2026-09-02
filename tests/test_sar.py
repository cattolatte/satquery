"""SAR rendering.

The bug this guards: dual-polarisation SAR carries VV and VH, and the previous
loader read only band 1 and replicated it to three channels. VH was discarded
entirely -- and VH is most of what separates vegetation from built-up in radar,
so the cross-modal tool was reading half its input.
"""
import numpy as np
import pytest

from satquery.tools.specialists import _sar_composite, _stretch


class TestStretch:
    def test_maps_into_unit_range(self):
        out = _stretch(np.random.default_rng(0).normal(50, 10, (32, 32)))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_constant_channel_does_not_divide_by_zero(self):
        out = _stretch(np.full((8, 8), 7.0))
        assert np.isfinite(out).all()

    def test_non_finite_values_are_survived(self):
        arr = np.full((8, 8), 3.0)
        arr[0, 0] = np.nan
        arr[0, 1] = np.inf
        assert np.isfinite(_stretch(arr)).all()

    def test_all_non_finite_returns_zeros(self):
        assert _stretch(np.full((4, 4), np.nan)).sum() == 0.0


class TestSARComposite:
    def test_dual_pol_uses_both_polarisations(self):
        """Changing VH alone must change the output. Under the old loader it
        could not, because VH was never read."""
        rng = np.random.default_rng(1)
        vv = rng.gamma(2.0, 1.0, (32, 32))
        vh_a = rng.gamma(2.0, 1.0, (32, 32))
        vh_b = rng.gamma(5.0, 1.0, (32, 32))
        a = _sar_composite([vv, vh_a])
        b = _sar_composite([vv, vh_b])
        assert not np.allclose(a, b), "VH is being ignored"

    def test_output_is_three_channels_in_unit_range(self):
        rng = np.random.default_rng(2)
        out = _sar_composite([rng.gamma(2.0, 1.0, (16, 16)),
                              rng.gamma(2.0, 1.0, (16, 16))])
        assert out.shape == (16, 16, 3)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_third_channel_is_the_polarisation_ratio(self):
        """Scaling VV and VH together leaves the ratio channel unchanged."""
        rng = np.random.default_rng(3)
        vv = rng.gamma(2.0, 1.0, (24, 24))
        vh = rng.gamma(2.0, 1.0, (24, 24))
        base = _sar_composite([vv, vh])
        scaled = _sar_composite([vv * 4.0, vh * 4.0])
        assert np.allclose(base[..., 2], scaled[..., 2], atol=1e-5)

    def test_single_band_still_renders(self):
        out = _sar_composite([np.random.default_rng(4).gamma(2.0, 1.0, (12, 12))])
        assert out.shape == (12, 12, 3)
        assert np.allclose(out[..., 0], out[..., 1])

    def test_zero_amplitude_does_not_produce_nan(self):
        """SAR amplitude can be exactly zero; log10 must be guarded."""
        out = _sar_composite([np.zeros((8, 8)), np.zeros((8, 8))])
        assert np.isfinite(out).all()


def test_end_to_end_on_a_synthetic_dual_pol_geotiff():
    """The loader must route a 2-band product through the SAR path."""
    rasterio = pytest.importorskip("rasterio")
    import tempfile
    from pathlib import Path
    from satquery.schema import ImageMeta, Modality
    from satquery.tools.specialists import _open

    rng = np.random.default_rng(5)
    data = np.stack([rng.gamma(2.0, 1.0, (64, 64)),
                     rng.gamma(4.0, 1.0, (64, 64))]).astype("float32")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "s1_grd_vv_vh.tif"
        with rasterio.open(path, "w", driver="GTiff", height=64, width=64,
                           count=2, dtype="float32") as dst:
            dst.write(data)
        meta = ImageMeta(path=str(path), fmt="GeoTIFF", width=64, height=64,
                         bands=2, modality=Modality.SAR)
        img = _open(meta)
        assert img.size == (64, 64) and img.mode == "RGB"
        # A real composite has three distinct channels, not one replicated.
        arr = np.asarray(img)
        assert not np.array_equal(arr[..., 0], arr[..., 1])
