"""ชุดทดสอบสำหรับ app/processing/sharpener.py"""

import numpy as np
import pytest

from app.processing.sharpener import (
    SharpenMethod,
    UnsharpMaskParams,
    LaplacianParams,
    DetailEnhanceParams,
    apply_unsharp_mask,
    apply_laplacian,
    apply_detail_enhance,
    sharpen_image,
)


@pytest.fixture
def sample_bgr() -> np.ndarray:
    return np.random.randint(50, 200, (60, 60, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgra() -> np.ndarray:
    return np.random.randint(50, 200, (60, 60, 4), dtype=np.uint8)


def test_unsharp_mask_bgr(sample_bgr: np.ndarray) -> None:
    params = UnsharpMaskParams(radius=2.0, amount=1.5, threshold=0, luminance_only=False)
    res = apply_unsharp_mask(sample_bgr, params)
    assert res.shape == sample_bgr.shape
    assert res.dtype == np.uint8


def test_unsharp_mask_luminance(sample_bgr: np.ndarray) -> None:
    params = UnsharpMaskParams(radius=2.0, amount=1.5, threshold=0, luminance_only=True)
    res = apply_unsharp_mask(sample_bgr, params)
    assert res.shape == sample_bgr.shape
    assert res.dtype == np.uint8


def test_unsharp_mask_bgra(sample_bgra: np.ndarray) -> None:
    params = UnsharpMaskParams(radius=1.5, amount=1.2, threshold=5, luminance_only=True)
    res = apply_unsharp_mask(sample_bgra, params)
    assert res.shape == (60, 60, 4)
    # ช่อง Alpha ต้องไม่เปลี่ยนแปลง
    np.testing.assert_array_equal(res[:, :, 3], sample_bgra[:, :, 3])


def test_laplacian_bgr(sample_bgr: np.ndarray) -> None:
    params = LaplacianParams(kernel_size=3, strength=0.5, luminance_only=True)
    res = apply_laplacian(sample_bgr, params)
    assert res.shape == sample_bgr.shape
    assert res.dtype == np.uint8


def test_laplacian_bgra(sample_bgra: np.ndarray) -> None:
    params = LaplacianParams(kernel_size=3, strength=0.4, luminance_only=False)
    res = apply_laplacian(sample_bgra, params)
    assert res.shape == (60, 60, 4)
    np.testing.assert_array_equal(res[:, :, 3], sample_bgra[:, :, 3])


def test_detail_enhance(sample_bgr: np.ndarray, sample_bgra: np.ndarray) -> None:
    params = DetailEnhanceParams(sigma_s=10.0, sigma_r=0.15)
    res_bgr = apply_detail_enhance(sample_bgr, params)
    assert res_bgr.shape == sample_bgr.shape

    res_bgra = apply_detail_enhance(sample_bgra, params)
    assert res_bgra.shape == sample_bgra.shape
    np.testing.assert_array_equal(res_bgra[:, :, 3], sample_bgra[:, :, 3])


def test_sharpen_image_intensity(sample_bgr: np.ndarray) -> None:
    params = UnsharpMaskParams(radius=2.0, amount=2.0)
    # Intensity 0.0 -> ต้องได้ภาพเดิม
    res_0 = sharpen_image(sample_bgr, SharpenMethod.UNSHARP_MASK, params, intensity=0.0)
    np.testing.assert_array_equal(res_0, sample_bgr)

    # Intensity 1.0 -> ต้องได้ผลลัพธ์ที่ปรับเต็มที่
    res_1 = sharpen_image(sample_bgr, SharpenMethod.UNSHARP_MASK, params, intensity=1.0)
    assert res_1.shape == sample_bgr.shape

    # Intensity 0.5 -> กึ่งกลาง
    res_half = sharpen_image(sample_bgr, SharpenMethod.UNSHARP_MASK, params, intensity=0.5)
    assert res_half.shape == sample_bgr.shape
