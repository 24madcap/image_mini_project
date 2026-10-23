"""โมเดลสถาปัตยกรรมโครงข่ายประสาทเทียม RRDBNet (Pure PyTorch)

รองรับ Real-ESRGAN x4plus (23 blocks, scale 4),
Real-ESRGAN x2plus (23 blocks, scale 2 พร้อม pixel-unshuffle),
และ Real-ESRGAN x4plus_anime_6B (6 blocks, scale 4)
ทำงานโดยตรงบน PyTorch ไม่ต้องพึ่งพา basicsr หรือไลบรารีภายนอก
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) 5 ชั้นคอนโวลูชัน"""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # residual scaling 0.2
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB) ประกอบด้วย 3 RDBs"""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """โครงสร้างเครือข่าย RRDBNet สำหรับ Super-Resolution

    Args:
        num_in_ch: จำนวน channel อินพุต (3 สำหรับ RGB)
        num_out_ch: จำนวน channel เอาต์พุต (3 สำหรับ RGB)
        scale: อัตราขยาย (2 หรือ 4)
        num_feat: จำนวน feature maps ภายใน (ค่าเริ่มต้น 64)
        num_block: จำนวน RRDB blocks (23 สำหรับ x4plus/x2plus, 6 สำหรับ anime_6B)
        num_grow_ch: จำนวนช่องที่เติบโตในแต่ละ Dense layer (ค่าเริ่มต้น 32)
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale

        # ถ้า scale = 2 ใช้ pixel-unshuffle ลดขนาด spatial 2x ทำให้ช่องสัญญาณเพิ่มเป็น 3 * 4 = 12
        actual_in_ch = num_in_ch * 4 if scale == 2 else num_in_ch

        self.conv_first = nn.Conv2d(actual_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling layers
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # ขั้นตอน Upsample ครั้งที่ 1 (2x)
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))

        # ขั้นตอน Upsample ครั้งที่ 2 (2x -> รวมเป็น 4x จากภาพย่อย หรือ 2x จากภาพต้นฉบับ)
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
