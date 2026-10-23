"""ยูทิลิตี้สำหรับจัดการไฟล์รูปภาพ"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# รูปแบบไฟล์ที่รองรับ
SUPPORTED_FORMATS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")

# ตัวกรองไฟล์สำหรับ dialog
FILE_FILTER: str = (
    "รูปภาพทั้งหมด (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;"
    "PNG (*.png);;"
    "JPEG (*.jpg *.jpeg);;"
    "BMP (*.bmp);;"
    "TIFF (*.tiff *.tif);;"
    "WebP (*.webp);;"
    "ไฟล์ทั้งหมด (*.*)"
)

SAVE_FILTER: str = (
    "PNG (*.png);;"
    "JPEG (*.jpg);;"
    "BMP (*.bmp);;"
    "TIFF (*.tiff);;"
    "WebP (*.webp)"
)


def is_supported_image(file_path: str) -> bool:
    """ตรวจสอบว่าไฟล์เป็นรูปภาพที่รองรับหรือไม่"""
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_FORMATS


def load_image_cv2(file_path: str) -> Optional[np.ndarray]:
    """โหลดรูปภาพด้วย OpenCV (BGR format)
    
    รองรับ path ภาษาไทยด้วยการใช้ numpy + imdecode
    """
    try:
        # ใช้ numpy เพื่อรองรับ path ที่มีอักขระพิเศษ (เช่น ภาษาไทย)
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        # แปลง BGRA → BGR ถ้าจำเป็น (เก็บ alpha channel ไว้)
        if len(img.shape) == 2:
            # ภาพ grayscale → แปลงเป็น BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # BGRA - เก็บไว้เป็น BGRA เพื่อรักษา transparency
            pass

        return img
    except Exception:
        return None


def save_image_cv2(
    img: np.ndarray,
    file_path: str,
    quality: int = 95,
) -> bool:
    """บันทึกรูปภาพด้วย OpenCV
    
    Args:
        img: ภาพในรูปแบบ numpy array (BGR หรือ BGRA)
        file_path: เส้นทางไฟล์ปลายทาง
        quality: คุณภาพ 1-100 (สำหรับ JPEG/WebP)
    
    Returns:
        True ถ้าบันทึกสำเร็จ
    """
    try:
        ext = Path(file_path).suffix.lower()

        # ตั้งค่าพารามิเตอร์การบันทึกตามรูปแบบไฟล์
        params: list[int] = []
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == ".png":
            # PNG compression level: 0 (ไม่บีบอัด) ถึง 9 (บีบอัดมากสุด)
            compression = max(0, min(9, (100 - quality) * 9 // 100))
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]

        # ใช้ imencode + tofile เพื่อรองรับ path ภาษาไทย
        success, buf = cv2.imencode(ext, img, params)
        if success:
            buf.tofile(file_path)
            return True
        return False
    except Exception:
        return False


def get_image_info(file_path: str) -> dict[str, any]:
    """ดึงข้อมูลรูปภาพ"""
    info: dict[str, any] = {
        "path": file_path,
        "filename": Path(file_path).name,
        "format": Path(file_path).suffix.upper().lstrip("."),
        "size_bytes": 0,
        "width": 0,
        "height": 0,
        "channels": 0,
    }

    try:
        info["size_bytes"] = os.path.getsize(file_path)

        img = load_image_cv2(file_path)
        if img is not None:
            info["height"], info["width"] = img.shape[:2]
            info["channels"] = img.shape[2] if len(img.shape) > 2 else 1
    except Exception:
        pass

    return info


def format_file_size(size_bytes: int) -> str:
    """แปลงขนาดไฟล์เป็นข้อความที่อ่านง่าย"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_images_in_folder(folder_path: str) -> list[str]:
    """ดึงรายการไฟล์รูปภาพทั้งหมดในโฟลเดอร์"""
    images: list[str] = []
    try:
        for entry in sorted(os.listdir(folder_path)):
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path) and is_supported_image(full_path):
                images.append(full_path)
    except Exception:
        pass
    return images


def create_thumbnail(img: np.ndarray, max_size: int = 256) -> np.ndarray:
    """สร้าง thumbnail จากภาพ โดยรักษาสัดส่วน"""
    h, w = img.shape[:2]
    if h <= max_size and w <= max_size:
        return img.copy()

    scale = min(max_size / w, max_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
