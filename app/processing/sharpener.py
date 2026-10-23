"""โมเดลเพิ่มความคมชัดรูปภาพ (Sharpening Module)

รองรับ 3 วิธีหลัก:
1. Unsharp Mask - วิธีมาตรฐาน ปรับรัศมี, ความแรง, threshold ได้
2. Laplacian Sharpening - กรองหาขอบด้วย Laplacian operator
3. Detail Enhancement - ใช้ OpenCV edge-preserving filter

ความสามารถพิเศษ:
- รองรับโหมด Luminance Only (ปรับความคมชัดเฉพาะช่องความสว่างในระบบสี LAB)
  เพื่อป้องกันการเกิดสัญญาณรบกวนของสี (Color Fringing / Chrominance Noise)
- แยกและรักษาช่อง Alpha (ความโปร่งใส) อย่างสมบูรณ์
- ผสมผลลัพธ์ด้วยแถบเลื่อนความเข้ม (Intensity / Blend) 0-100%
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import cv2
import numpy as np


class SharpenMethod(Enum):
    """วิธีเพิ่มความคมชัด"""
    UNSHARP_MASK = "unsharp_mask"
    LAPLACIAN = "laplacian"
    DETAIL_ENHANCE = "detail_enhance"


@dataclass
class UnsharpMaskParams:
    """พารามิเตอร์สำหรับ Unsharp Mask"""
    radius: float = 2.0       # รัศมี Gaussian blur (sigma)
    amount: float = 1.5       # ความแรงตัวคูณความคมชัด
    threshold: int = 0        # ค่าขีดจำกัดความต่างต่ำสุดที่จะเพิ่มความคมชัด (0-255)
    luminance_only: bool = True  # ปรับเฉพาะช่องความสว่างในระบบ LAB


@dataclass
class LaplacianParams:
    """พารามิเตอร์สำหรับ Laplacian"""
    kernel_size: int = 3      # ขนาด kernel (1, 3, 5, 7)
    strength: float = 0.5     # ความแรง (0.05 - 2.0)
    luminance_only: bool = True


@dataclass
class DetailEnhanceParams:
    """พารามิเตอร์สำหรับ Detail Enhancement"""
    sigma_s: float = 10.0     # ความเรียบเนียนเชิงพื้นที่ (0-200)
    sigma_r: float = 0.15     # ความเรียบเนียนเชิงสี (0-1)


def _apply_on_bgr_or_lab(
    bgr_img: np.ndarray,
    filter_func,
    luminance_only: bool = True,
) -> np.ndarray:
    """ฟังก์ชันช่วยเหลือสำหรับนำตัวกรองไปใช้กับภาพ BGR หรือช่อง L ของ LAB"""
    if not luminance_only:
        return filter_func(bgr_img)

    # แปลงเป็น LAB แล้วปรับเฉพาะ L channel
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # ปรับแต่งเฉพาะช่อง L
    sharpened_l = filter_func(l_channel)

    # รวมช่องสัญญาณและแปลงกลับเป็น BGR
    merged_lab = cv2.merge([sharpened_l, a_channel, b_channel])
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(
    img: np.ndarray,
    params: Optional[UnsharpMaskParams] = None,
) -> np.ndarray:
    """เพิ่มความคมชัดด้วย Unsharp Mask

    สูตร: Sharpened = Original + (Original - Blurred) * amount
    """
    if params is None:
        params = UnsharpMaskParams()

    # แยก Alpha channel ถ้ามี
    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    if has_alpha:
        work_img = img[:, :, :3]
        alpha = img[:, :, 3:]
    else:
        work_img = img
        alpha = None

    def _unsharp_core(channel_or_bgr: np.ndarray) -> np.ndarray:
        ksize = int(params.radius * 4) + 1
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, ksize)

        f_img = channel_or_bgr.astype(np.float64)
        blurred = cv2.GaussianBlur(f_img, (ksize, ksize), params.radius)
        mask = f_img - blurred

        if params.threshold > 0:
            low_contrast = np.abs(mask) < params.threshold
            mask[low_contrast] = 0

        res = f_img + mask * params.amount
        return np.clip(res, 0, 255).astype(np.uint8)

    # ประมวลผล
    if len(work_img.shape) == 3 and params.luminance_only:
        processed_bgr = _apply_on_bgr_or_lab(work_img, _unsharp_core, luminance_only=True)
    else:
        processed_bgr = _unsharp_core(work_img)

    if has_alpha:
        return np.concatenate([processed_bgr, alpha], axis=2)
    return processed_bgr


def apply_laplacian(
    img: np.ndarray,
    params: Optional[LaplacianParams] = None,
) -> np.ndarray:
    """เพิ่มความคมชัดด้วยตัวกรอง Laplacian เพื่อหาขอบและผสานกลับ"""
    if params is None:
        params = LaplacianParams()

    ksize = params.kernel_size
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(1, min(31, ksize))

    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    if has_alpha:
        work_img = img[:, :, :3]
        alpha = img[:, :, 3:]
    else:
        work_img = img
        alpha = None

    def _laplacian_core(channel_or_bgr: np.ndarray) -> np.ndarray:
        f_img = channel_or_bgr.astype(np.float64)
        lap = cv2.Laplacian(f_img, cv2.CV_64F, ksize=ksize)
        res = f_img - lap * params.strength
        return np.clip(res, 0, 255).astype(np.uint8)

    if len(work_img.shape) == 3 and params.luminance_only:
        processed_bgr = _apply_on_bgr_or_lab(work_img, _laplacian_core, luminance_only=True)
    else:
        processed_bgr = _laplacian_core(work_img)

    if has_alpha:
        return np.concatenate([processed_bgr, alpha], axis=2)
    return processed_bgr


def apply_detail_enhance(
    img: np.ndarray,
    params: Optional[DetailEnhanceParams] = None,
) -> np.ndarray:
    """เพิ่มรายละเอียดภาพด้วย OpenCV detailEnhance (Edge-preserving filter)"""
    if params is None:
        params = DetailEnhanceParams()

    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    if has_alpha:
        work_img = img[:, :, :3]
        alpha = img[:, :, 3:]
    elif len(img.shape) == 2:
        work_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = None
    else:
        work_img = img
        alpha = None

    enhanced = cv2.detailEnhance(
        work_img,
        sigma_s=params.sigma_s,
        sigma_r=params.sigma_r,
    )

    if has_alpha:
        return np.concatenate([enhanced, alpha], axis=2)
    return enhanced


def sharpen_image(
    img: np.ndarray,
    method: SharpenMethod,
    params: Union[UnsharpMaskParams, LaplacianParams, DetailEnhanceParams, None] = None,
    intensity: float = 1.0,
) -> np.ndarray:
    """ฟังก์ชันหลักสำหรับเพิ่มความคมชัดรูปภาพ

    Args:
        img: ภาพอินพุต (numpy array BGR หรือ BGRA)
        method: วิธีการเพิ่มความคมชัด
        params: พารามิเตอร์สำหรับวิธีที่เลือก
        intensity: ความเข้มของการผสมภาพ 0.0 (ภาพเดิม) ถึง 1.0 (ปรับเต็มที่)

    Returns:
        ภาพที่ผ่านการเพิ่มความคมชัดแล้ว
    """
    if intensity <= 0.0:
        return img.copy()

    if method == SharpenMethod.UNSHARP_MASK:
        sharpened = apply_unsharp_mask(img, params if isinstance(params, UnsharpMaskParams) else None)
    elif method == SharpenMethod.LAPLACIAN:
        sharpened = apply_laplacian(img, params if isinstance(params, LaplacianParams) else None)
    elif method == SharpenMethod.DETAIL_ENHANCE:
        sharpened = apply_detail_enhance(img, params if isinstance(params, DetailEnhanceParams) else None)
    else:
        return img.copy()

    # ผสมภาพตาม intensity
    if intensity < 1.0:
        has_alpha = len(img.shape) == 3 and img.shape[2] == 4
        if has_alpha:
            bgr_orig = img[:, :, :3]
            bgr_sharp = sharpened[:, :, :3]
            blended_bgr = cv2.addWeighted(bgr_orig, 1.0 - intensity, bgr_sharp, intensity, 0)
            return np.concatenate([blended_bgr, img[:, :, 3:]], axis=2)
        else:
            return cv2.addWeighted(img, 1.0 - intensity, sharpened, intensity, 0)

    return sharpened
