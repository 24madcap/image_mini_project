"""End-to-End Pipeline Integration Test"""

import os
import tempfile
import numpy as np
import pytest

from app.processing.upscaler import UpscaleMode, UpscaleParams, upscale_image
from app.processing.sharpener import SharpenMethod, UnsharpMaskParams, sharpen_image
from app.utils.file_utils import save_image_cv2, load_image_cv2


def test_full_pipeline_traditional_and_sharpen() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input_test.png")
        output_file = os.path.join(tmpdir, "output_test.png")

        # 1. สร้างภาพต้นฉบับ
        orig = np.random.randint(50, 200, (40, 50, 3), dtype=np.uint8)
        assert save_image_cv2(orig, input_file) is True

        # 2. โหลดภาพ
        loaded = load_image_cv2(input_file)
        assert loaded is not None

        # 3. ขยายขนาดภาพ 2x ด้วย Lanczos
        params = UpscaleParams(scale=2.0)
        upscaled = upscale_image(loaded, params)
        assert upscaled.shape == (80, 100, 3)

        # 4. เพิ่มความคมชัดด้วย Unsharp Mask (Luminance Only)
        sharpen_params = UnsharpMaskParams(radius=2.0, amount=1.5, luminance_only=True)
        sharpened = sharpen_image(upscaled, SharpenMethod.UNSHARP_MASK, sharpen_params, intensity=1.0)
        assert sharpened.shape == (80, 100, 3)

        # 5. บันทึกผลลัพธ์
        assert save_image_cv2(sharpened, output_file, quality=95) is True
        assert os.path.exists(output_file)

        final_img = load_image_cv2(output_file)
        assert final_img is not None
        assert final_img.shape == (80, 100, 3)


def test_full_pipeline_ai_and_sharpen() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input_ai.png")
        output_file = os.path.join(tmpdir, "output_ai.jpg")

        orig = np.random.randint(40, 220, (32, 32, 3), dtype=np.uint8)
        assert save_image_cv2(orig, input_file) is True

        loaded = load_image_cv2(input_file)
        assert loaded is not None

        # AI 4x (Anime model)
        params = UpscaleParams(
            mode=UpscaleMode.AI_REALESRGAN,
            scale=4.0,
            ai_model_name="RealESRGAN_x4plus_anime_6B",
        )
        upscaled = upscale_image(loaded, params)
        assert upscaled.shape == (128, 128, 3)

        # เพิ่มความคมชัด
        sharpen_params = UnsharpMaskParams(radius=1.5, amount=1.2, luminance_only=True)
        sharpened = sharpen_image(upscaled, SharpenMethod.UNSHARP_MASK, sharpen_params, intensity=0.8)
        assert sharpened.shape == (128, 128, 3)

        # บันทึกเป็น JPEG
        assert save_image_cv2(sharpened, output_file, quality=90) is True
        assert os.path.exists(output_file)
