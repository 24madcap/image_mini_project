"""ชุดทดสอบสำหรับ app/utils/file_utils.py"""

import os
import tempfile
import numpy as np
import pytest

from app.utils.file_utils import (
    is_supported_image,
    load_image_cv2,
    save_image_cv2,
    get_image_info,
    format_file_size,
    create_thumbnail,
)


def test_is_supported_image() -> None:
    assert is_supported_image("test.png") is True
    assert is_supported_image("test.JPG") is True
    assert is_supported_image("test.jpeg") is True
    assert is_supported_image("test.webp") is True
    assert is_supported_image("test.bmp") is True
    assert is_supported_image("test.tiff") is True
    assert is_supported_image("test.txt") is False
    assert is_supported_image("test.pdf") is False


def test_format_file_size() -> None:
    assert format_file_size(500) == "500 B"
    assert format_file_size(2048) == "2.0 KB"
    assert format_file_size(5 * 1024 * 1024) == "5.0 MB"


def test_save_and_load_image_bgr() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # ทดสอบ path ที่มีภาษาไทย
        file_path = os.path.join(tmpdir, "ทดสอบ_รูปภาพ.png")

        # สร้างภาพ BGR สุ่ม
        orig = np.random.randint(0, 256, (50, 60, 3), dtype=np.uint8)
        success = save_image_cv2(orig, file_path)
        assert success is True
        assert os.path.exists(file_path)

        loaded = load_image_cv2(file_path)
        assert loaded is not None
        assert loaded.shape == (50, 60, 3)
        # ตรวจสอบค่าพิกเซลตรงกัน (PNG เป็น lossless)
        np.testing.assert_array_equal(orig, loaded)


def test_save_and_load_image_bgra() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "transparent_ภาพ.png")

        # สร้างภาพ BGRA 4 channels พร้อม alpha
        orig = np.random.randint(0, 256, (40, 40, 4), dtype=np.uint8)
        success = save_image_cv2(orig, file_path)
        assert success is True

        loaded = load_image_cv2(file_path)
        assert loaded is not None
        assert loaded.shape == (40, 40, 4)
        np.testing.assert_array_equal(orig, loaded)


def test_get_image_info() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "info_test.jpg")
        img = np.ones((80, 100, 3), dtype=np.uint8) * 128
        save_image_cv2(img, file_path, quality=90)

        info = get_image_info(file_path)
        assert info["width"] == 100
        assert info["height"] == 80
        assert info["channels"] == 3
        assert info["size_bytes"] > 0


def test_create_thumbnail() -> None:
    img = np.zeros((400, 800, 3), dtype=np.uint8)
    thumb = create_thumbnail(img, max_size=200)
    h, w = thumb.shape[:2]
    assert w <= 200
    assert h <= 200
    assert w == 200
    assert h == 100
