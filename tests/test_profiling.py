"""
Tests for the timing summary and image listing in rasher_qc.profiling.

The timing itself depends on hardware, so only the arithmetic that turns
timings into reported numbers is tested, using hand-calculated values.
"""

import pandas as pd
import pytest

from rasher_qc.profiling import list_images, summarise


def fake_timings():
    """Two images with easy numbers: stage totals of 10 ms and 20 ms."""
    return pd.DataFrame({
        "preprocess_ms": [1.0, 3.0],     # mean 2
        "inference_ms": [4.0, 6.0],      # mean 5
        "postprocess_ms": [5.0, 11.0],   # mean 8
        "wall_ms": [10.0, 30.0],         # mean 20
        "n_boxes": [1, 3],
    })


def test_summary_means_and_rate():
    s = summarise(fake_timings())
    assert s["preprocess_ms"] == 2.0
    assert s["inference_ms"] == 5.0
    assert s["postprocess_ms"] == 8.0
    assert s["wall_ms_mean"] == 20.0
    assert s["images_per_s"] == pytest.approx(50.0)     # 1000 ms / 20 ms
    assert s["boxes_per_image"] == 2.0


def test_stage_shares_sum_to_one():
    s = summarise(fake_timings())
    assert s["postprocess_share"] == pytest.approx(8 / 15)
    assert s["preprocess_share"] + s["inference_share"] + s["postprocess_share"] == \
        pytest.approx(1.0)


def test_achieved_throughput():
    # 8 GFLOP in a mean inference time of 5 ms = 1,600 GFLOP/s
    assert summarise(fake_timings(), gflops=8.0)["achieved_gflop_per_s"] == \
        pytest.approx(1600.0)
    assert summarise(fake_timings())["achieved_gflop_per_s"] is None


def test_list_images_filters_and_sorts(tmp_path):
    for name in ("b.jpg", "a.PNG", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.PNG", "b.jpg"]


def test_list_images_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no images"):
        list_images(tmp_path)
