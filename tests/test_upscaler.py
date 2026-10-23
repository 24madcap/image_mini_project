"""ชุดทดสอบสำหรับ app/processing/upscaler.py และ ai_engine.py"""

import numpy as np
import pytest

from app.processing.upscaler import (
    UpscaleMode,
    InterpolationMethod,
    UpscaleParams,
    upscale_traditional,
    upscale_ai,
    upscale_image,
    check_ai_available,
)
from app.processing.ai_engine import AVAILABLE_MODELS, is_model_downloaded


@pytest.fixture
def sample_bgr() -> np.ndarray:
    return np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgra() -> np.ndarray:
    return np.random.randint(0, 256, (32, 48, 4), dtype=np.uint8)


def test_traditional_upscale_scale(sample_bgr: np.ndarray) -> None:
    # 2x scale
    params = UpscaleParams(
        mode=UpscaleMode.TRADITIONAL,
        scale=2.0,
        interpolation=InterpolationMethod.LANCZOS,
    )
    res = upscale_traditional(sample_bgr, params)
    assert res.shape == (64, 96, 3)

    # 1.5x scale
    params_15 = UpscaleParams(
        mode=UpscaleMode.TRADITIONAL,
        scale=1.5,
        interpolation=InterpolationMethod.BICUBIC,
    )
    res_15 = upscale_traditional(sample_bgr, params_15)
    assert res_15.shape == (48, 72, 3)


def test_traditional_upscale_custom_size(sample_bgr: np.ndarray) -> None:
    params = UpscaleParams(
        mode=UpscaleMode.TRADITIONAL,
        use_custom_size=True,
        custom_width=200,
        custom_height=150,
        interpolation=InterpolationMethod.BILINEAR,
    )
    res = upscale_traditional(sample_bgr, params)
    assert res.shape == (150, 200, 3)


def test_traditional_all_interpolation_methods(sample_bgr: np.ndarray) -> None:
    for method in InterpolationMethod:
        params = UpscaleParams(
            mode=UpscaleMode.TRADITIONAL,
            scale=2.0,
            interpolation=method,
        )
        res = upscale_traditional(sample_bgr, params)
        assert res.shape == (64, 96, 3)


def test_check_ai_available() -> None:
    available, msg = check_ai_available()
    assert available is True
    assert "AI พร้อมใช้งาน" in msg


def test_ai_upscale_anime(sample_bgr: np.ndarray) -> None:
    # ทดสอบโมเดล RealESRGAN_x4plus_anime_6B (ขยาย 4 เท่า)
    params = UpscaleParams(
        mode=UpscaleMode.AI_REALESRGAN,
        scale=4.0,
        ai_model_name="RealESRGAN_x4plus_anime_6B",
    )
    res = upscale_ai(sample_bgr, params)
    assert res.shape == (128, 192, 3)


def test_ai_upscale_with_custom_scale(sample_bgr: np.ndarray) -> None:
    # ทดสอบกรณีใช้โมเดล AI ร่วมกับการปรับขนาดเป็น 2x
    params = UpscaleParams(
        mode=UpscaleMode.AI_REALESRGAN,
        scale=2.0,
        ai_model_name="RealESRGAN_x2plus",
    )
    res = upscale_ai(sample_bgr, params)
    assert res.shape == (64, 96, 3)


def test_ai_upscale_bgra(sample_bgra: np.ndarray) -> None:
    params = UpscaleParams(
        mode=UpscaleMode.AI_REALESRGAN,
        scale=2.0,
        ai_model_name="RealESRGAN_x2plus",
    )
    res = upscale_ai(sample_bgra, params)
    assert res.shape == (64, 96, 4)
