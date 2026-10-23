"""เครื่องมือประมวลผล AI Super Resolution (AI Engine)

จัดการการโหลดโมเดล, แคช, การดาวน์โหลดอัตโนมัติ,
และการขยายภาพแบบแบ่งเป็นส่วนย่อย (Tiled Inference) เพื่อประหยัดหน่วยความจำ
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from app.processing.models.rrdbnet import RRDBNet


@dataclass
class ModelInfo:
    """ข้อมูลรายละเอียดของโมเดล AI"""
    name: str
    display_name: str
    scale: int
    num_blocks: int
    filename: str
    url: str
    description: str


# รายการโมเดลที่รองรับ
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    "RealESRGAN_x4plus": ModelInfo(
        name="RealESRGAN_x4plus",
        display_name="Real-ESRGAN x4plus (ภาพถ่ายทั่วไป 4x)",
        scale=4,
        num_blocks=23,
        filename="RealESRGAN_x4plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        description="เหมาะสำหรับภาพถ่ายทั่วไป ความคมชัดสูงสุด ขยาย 4 เท่า",
    ),
    "RealESRGAN_x4plus_anime_6B": ModelInfo(
        name="RealESRGAN_x4plus_anime_6B",
        display_name="Real-ESRGAN Anime 6B (ภาพการ์ตูน/อนิเมะ 4x)",
        scale=4,
        num_blocks=6,
        filename="RealESRGAN_x4plus_anime_6B.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        description="ปรับแต่งเป็นพิเศษสำหรับภาพวาด ภาพการ์ตูน ขนาดเล็กและประมวลผลเร็วกว่า",
    ),
    "RealESRGAN_x2plus": ModelInfo(
        name="RealESRGAN_x2plus",
        display_name="Real-ESRGAN x2plus (ภาพถ่ายทั่วไป 2x)",
        scale=2,
        num_blocks=23,
        filename="RealESRGAN_x2plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        description="เหมาะสำหรับขยายภาพ 2 เท่า ให้เส้นขอบที่คมชัดและเป็นธรรมชาติ",
    ),
}

# แคชเก็บโมเดลที่โหลดไว้แล้วในหน่วยความจำ (หลีกเลี่ยงการอ่านไฟล์ซ้ำ)
_LOADED_MODELS: dict[str, torch.nn.Module] = {}


def get_models_dir() -> str:
    """ดึงโฟลเดอร์สำหรับเก็บไฟล์โมเดล .pth"""
    # ตรวจสอบว่ามีโฟลเดอร์ models ใน root หรือ app data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def check_ai_status() -> tuple[bool, str]:
    """ตรวจสอบสถานะความพร้อมของระบบ AI"""
    try:
        device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        threads = torch.get_num_threads()
        return True, f"ระบบ AI พร้อมใช้งานบน {device_name} (Threads: {threads})"
    except Exception as e:
        return False, f"ระบบ AI ไม่พร้อมใช้งาน: {e}"


def get_model_path(model_name: str) -> str:
    """ดึง path ของไฟล์โมเดล"""
    info = AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS["RealESRGAN_x4plus"])
    return os.path.join(get_models_dir(), info.filename)


def is_model_downloaded(model_name: str) -> bool:
    """ตรวจสอบว่าไฟล์โมเดลถูกดาวน์โหลดมาไว้ในเครื่องแล้วหรือไม่"""
    path = get_model_path(model_name)
    return os.path.exists(path) and os.path.getsize(path) > 1_000_000


def download_model(
    model_name: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> str:
    """ดาวน์โหลดโมเดลอัตโนมัติหากยังไม่มีในเครื่อง"""
    info = AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS["RealESRGAN_x4plus"])
    target_path = get_model_path(model_name)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1_000_000:
        return target_path

    if progress_callback:
        progress_callback(0, f"กำลังดาวน์โหลดโมเดล {info.filename}...")

    req = urllib.request.Request(info.url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 256 * 1024

        temp_path = target_path + ".tmp"
        with open(temp_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and progress_callback:
                    pct = int((downloaded / total) * 100)
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total / (1024 * 1024)
                    progress_callback(pct, f"กำลังดาวน์โหลดโมเดล: {mb_cur:.1f}/{mb_tot:.1f} MB ({pct}%)")

        # ย้ายไฟล์ชั่วคราวเมื่อดาวน์โหลดเสร็จสมบูรณ์
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)

    if progress_callback:
        progress_callback(100, "ดาวน์โหลดโมเดลเสร็จสิ้น")

    return target_path


def load_ai_model(
    model_name: str,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> tuple[torch.nn.Module, ModelInfo, torch.device]:
    """โหลดโมเดล RRDBNet เข้าสู่หน่วยความจำ"""
    if model_name not in AVAILABLE_MODELS:
        model_name = "RealESRGAN_x4plus"

    info = AVAILABLE_MODELS[model_name]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = f"{model_name}_{device.type}"
    if cache_key in _LOADED_MODELS:
        return _LOADED_MODELS[cache_key], info, device

    # ตรวจสอบและดาวน์โหลดหากจำเป็น
    model_path = get_model_path(model_name)
    if not is_model_downloaded(model_name):
        download_model(model_name, progress_callback)

    if progress_callback:
        progress_callback(5, f"กำลังโหลดโมเดล {info.name} เข้าสู่หน่วยความจำ...")

    # สร้างโครงข่าย RRDBNet
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        scale=info.scale,
        num_feat=64,
        num_block=info.num_blocks,
        num_grow_ch=32,
    )

    # โหลด weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    # แคชโมเดลไว้
    _LOADED_MODELS[cache_key] = model

    return model, info, device


def run_tiled_ai_upscale(
    img: np.ndarray,
    model_name: str = "RealESRGAN_x4plus",
    tile_size: int = 400,
    tile_pad: int = 12,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """ขยายภาพด้วย AI พร้อมเทคนิค Tiled Inference ป้องกันปัญหาหน่วยความจำไม่พอ

    Args:
        img: ภาพอินพุต (BGR หรือ BGRA แบบ numpy array uint8)
        model_name: ชื่อโมเดล AI ที่ต้องการใช้
        tile_size: ขนาดของแต่ละบล็อกที่จะประมวลผล
        tile_pad: ขนาดระยะขอบซ้อนทับเพื่อลบรอยต่อระหว่างบล็อก
        progress_callback: callback รายงาน (เปอร์เซ็นต์, ข้อความ)
        is_cancelled: callback ตรวจสอบว่าผู้ใช้กดยกเลิกหรือไม่

    Returns:
        ภาพที่ผ่านการขยายขนาดแล้ว
    """
    if is_cancelled and is_cancelled():
        return img

    # ตรวจสอบ channel (รองรับทั้ง BGR และ BGRA)
    has_alpha = len(img.shape) == 3 and img.shape[2] == 4
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    elif len(img.shape) == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = None
    else:
        bgr = img
        alpha = None

    h, w, c = bgr.shape

    # โหลดโมเดล
    model, info, device = load_ai_model(model_name, progress_callback=progress_callback)
    scale = info.scale

    if progress_callback:
        progress_callback(10, f"เตรียมการประมวลผล AI ({info.display_name})...")

    # แปลงภาพ BGR (0-255) เป็น RGB Tensor (0.0-1.0)
    # PyTorch Real-ESRGAN คาดหวัง RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor_img = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
        / 255.0
    )

    out_h, out_w = h * scale, w * scale
    output_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # คำนวณจำนวน tile
    tiles_x = max(1, (w + tile_size - 1) // tile_size)
    tiles_y = max(1, (h + tile_size - 1) // tile_size)
    total_tiles = tiles_x * tiles_y

    processed_tiles = 0

    for yi in range(tiles_y):
        for xi in range(tiles_x):
            if is_cancelled and is_cancelled():
                return img

            x1 = xi * tile_size
            y1 = yi * tile_size
            x2 = min(x1 + tile_size, w)
            y2 = min(y1 + tile_size, h)

            # ขยายขอบพร้อม padding
            pad_left = min(tile_pad, x1)
            pad_top = min(tile_pad, y1)
            pad_right = min(tile_pad, w - x2)
            pad_bottom = min(tile_pad, h - y2)

            in_x1 = x1 - pad_left
            in_y1 = y1 - pad_top
            in_x2 = x2 + pad_right
            in_y2 = y2 + pad_bottom

            tile = tensor_img[:, :, in_y1:in_y2, in_x1:in_x2]

            with torch.no_grad():
                out_tile = model(tile)

            # แปลงผลลัพธ์กลับเป็น numpy uint8
            out_np = out_tile.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            out_np = (out_np * 255.0).round().astype(np.uint8)

            # ตัดส่วน padding ออก
            crop_top = pad_top * scale
            crop_bottom = out_np.shape[0] - (pad_bottom * scale)
            crop_left = pad_left * scale
            crop_right = out_np.shape[1] - (pad_right * scale)

            cropped = out_np[crop_top:crop_bottom, crop_left:crop_right]

            # วางลงในภาพผลลัพธ์
            out_x1 = x1 * scale
            out_y1 = y1 * scale
            out_x2 = out_x1 + cropped.shape[1]
            out_y2 = out_y1 + cropped.shape[0]

            output_rgb[out_y1:out_y2, out_x1:out_x2] = cropped

            processed_tiles += 1
            if progress_callback:
                # แมพเปอร์เซ็นต์ช่วง 10 - 90%
                pct = 10 + int((processed_tiles / total_tiles) * 80)
                progress_callback(
                    pct,
                    f"กำลังประมวลผล AI บล็อก {processed_tiles}/{total_tiles} ({pct}%)",
                )

    # แปลง RGB กลับเป็น BGR
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    # จัดการ Alpha channel ถ้ามี
    if has_alpha:
        if progress_callback:
            progress_callback(92, "กำลังขยายช่อง Alpha (ความโปร่งใส)...")
        output_alpha = cv2.resize(
            alpha,
            (out_w, out_h),
            interpolation=cv2.INTER_LANCZOS4,
        )
        result = np.dstack([output_bgr, output_alpha])
    else:
        result = output_bgr

    if progress_callback:
        progress_callback(95, "เสร็จสิ้นขั้นตอน AI Super-Resolution")

    return result
