"""วิดเจ็ตแสดงและเปรียบเทียบภาพ (Image Viewer) ระดับมืออาชีพ

คุณสมบัติ:
- โหมดเปรียบเทียบ 3 รูปแบบ:
  1. ตัวเลื่อนเปรียบเทียบซ้อนทับ (Split Slider ◀▶) โดยปรับสเกลให้ตรงกัน 100%
  2. แสดงคู่กันซ้าย-ขวา (Side-by-Side) พร้อมการซูมและเลื่อนพร้อมกัน
  3. แสดงภาพเดี่ยว (Single Image)
- ซูมเข้า/ออกอย่างนุ่มนวล โดยอิงจุดกึ่งกลางที่ตำแหน่งเมาส์ (Mouse-anchored Zoom)
- เลื่อนดูภาพ (Pan) ด้วยการคลิกลาก หรือกด Spacebar ค้างไว้
- Pixel Inspector HUD แสดงพิกัด (X, Y) และค่าสี RGB ของพิกเซลใต้เคอร์เซอร์เมาส์
- แถบป้ายกำกับ "ก่อน" และ "หลัง" ที่สวยงาม
"""

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush, QWheelEvent,
    QMouseEvent, QPaintEvent, QResizeEvent, QFont, QKeyEvent,
)
from PyQt6.QtWidgets import QWidget


def cv2_to_qimage(img: np.ndarray) -> QImage:
    """แปลง OpenCV BGR/BGRA array เป็น QImage"""
    if img is None:
        return QImage()

    if len(img.shape) == 2:
        h, w = img.shape
        return QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    elif img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    elif img.shape[2] == 4:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        h, w, ch = rgba.shape
        bytes_per_line = ch * w
        return QImage(rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888).copy()

    return QImage()


class ImageViewer(QWidget):
    """วิดเจ็ตแสดงภาพและเปรียบเทียบผลลัพธ์"""

    # โหมดการแสดงผล
    MODE_COMPARE_SLIDER = "compare_slider"  # สไลเดอร์แบ่งครึ่ง
    MODE_SIDE_BY_SIDE = "side_by_side"      # คู่กันซ้าย-ขวา
    MODE_SINGLE = "single"                  # ภาพเดี่ยว

    # สัญญาณแจ้งเมื่อระดับซูมเปลี่ยน
    zoom_changed = pyqtSignal(int)
    pixel_hovered = pyqtSignal(int, int, int, int, int)  # x, y, r, g, b

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # ข้อมูลภาพ
        self._before_img: Optional[np.ndarray] = None
        self._after_img: Optional[np.ndarray] = None
        self._before_qimg: Optional[QImage] = None
        self._after_qimg: Optional[QImage] = None

        # สถานะการแสดงผล
        self._mode: str = self.MODE_SINGLE
        self._zoom: float = 1.0
        self._min_zoom: float = 0.02
        self._max_zoom: float = 30.0
        self._offset = QPointF(0, 0)

        # การเลื่อนเปรียบเทียบ (Split Slider)
        self._slider_pos: float = 0.5  # 0.0 ถึง 1.0
        self._slider_dragging: bool = False
        self._slider_width: int = 2

        # การเลื่อนดูภาพ (Pan)
        self._panning: bool = False
        self._space_pressed: bool = False
        self._pan_start: QPoint = QPoint()
        self._pan_offset_start: QPointF = QPointF()

        # พิกัดเคอร์เซอร์และค่าสีสำหรับ HUD
        self._cursor_pos: Optional[QPoint] = None
        self._hover_pixel: Optional[tuple[int, int, int, int, int]] = None  # img_x, img_y, r, g, b

        self.setStyleSheet("background-color: #1a1a1a;")

    # ──────────────────────────────────────
    # Public API
    # ──────────────────────────────────────
    def set_before_image(self, img: Optional[np.ndarray]) -> None:
        """ตั้งภาพต้นฉบับก่อนประมวลผล"""
        self._before_img = img
        self._before_qimg = cv2_to_qimage(img) if img is not None else None
        self._fit_image()
        self.update()

    def set_after_image(self, img: Optional[np.ndarray]) -> None:
        """ตั้งภาพผลลัพธ์หลังประมวลผล"""
        self._after_img = img
        if img is not None:
            self._after_qimg = cv2_to_qimage(img)
            # สลับไปโหมดเปรียบเทียบอัตโนมัติหากยังเป็นโหมดภาพเดี่ยว
            if self._mode == self.MODE_SINGLE and self._before_qimg is not None:
                self._mode = self.MODE_COMPARE_SLIDER
        else:
            self._after_qimg = None
            self._mode = self.MODE_SINGLE
        self.update()

    def set_view_mode(self, mode: str) -> None:
        """เปลี่ยนโหมดการแสดงผล"""
        if mode in (self.MODE_COMPARE_SLIDER, self.MODE_SIDE_BY_SIDE, self.MODE_SINGLE):
            self._mode = mode
            self.update()

    def clear(self) -> None:
        """ล้างภาพทั้งหมด"""
        self._before_img = None
        self._after_img = None
        self._before_qimg = None
        self._after_qimg = None
        self._mode = self.MODE_SINGLE
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._hover_pixel = None
        self.update()

    def zoom_in(self) -> None:
        """ซูมเข้า 25%"""
        self._set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        """ซูมออก 25%"""
        self._set_zoom(self._zoom / 1.25)

    def zoom_fit(self) -> None:
        """ปรับขนาดภาพให้พอดีหน้าต่าง"""
        self._fit_image()
        self.update()

    def zoom_actual(self) -> None:
        """ซูมขนาดจริง 100%"""
        ref_w, ref_h = self._get_ref_size()
        if ref_w > 0 and ref_h > 0:
            self._zoom = 1.0
            self._offset = QPointF(
                (self.width() - ref_w) / 2,
                (self.height() - ref_h) / 2,
            )
            self.zoom_changed.emit(100)
            self.update()

    # ──────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────
    def _get_ref_size(self) -> tuple[int, int]:
        """ดึงขนาดอ้างอิงของภาพที่กำลังแสดงผล (ใช้ภาพ After ถ้ามี มิฉะนั้นใช้ Before)"""
        if self._after_qimg is not None and not self._after_qimg.isNull():
            return self._after_qimg.width(), self._after_qimg.height()
        elif self._before_qimg is not None and not self._before_qimg.isNull():
            return self._before_qimg.width(), self._before_qimg.height()
        return 0, 0

    def _fit_image(self) -> None:
        """คำนวณและปรับซูมให้ภาพพอดีกับพื้นที่แสดงผล"""
        ref_w, ref_h = self._get_ref_size()
        if ref_w <= 0 or ref_h <= 0:
            return

        vw = max(50, self.width() - 30)
        vh = max(50, self.height() - 30)

        if self._mode == self.MODE_SIDE_BY_SIDE and self._after_qimg is not None:
            # ในโหมดคู่กัน แต่ละข้างได้พื้นที่ครึ่งหนึ่งของความกว้าง
            vw = max(50, (self.width() - 40) / 2)

        scale = min(vw / ref_w, vh / ref_h)
        self._zoom = max(self._min_zoom, min(self._max_zoom, scale))

        if self._mode == self.MODE_SIDE_BY_SIDE and self._after_qimg is not None:
            half_w = self.width() / 2
            self._offset = QPointF(
                (half_w - ref_w * self._zoom) / 2,
                (self.height() - ref_h * self._zoom) / 2,
            )
        else:
            self._offset = QPointF(
                (self.width() - ref_w * self._zoom) / 2,
                (self.height() - ref_h * self._zoom) / 2,
            )

        self.zoom_changed.emit(int(self._zoom * 100))

    def _set_zoom(self, new_zoom: float, center_point: Optional[QPointF] = None) -> None:
        """ตั้งค่าระดับการซูมพร้อมรักษาจุดกึ่งกลาง"""
        old_zoom = self._zoom
        clamped_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))

        if old_zoom != clamped_zoom:
            center = center_point if center_point is not None else QPointF(self.width() / 2, self.height() / 2)
            ratio = clamped_zoom / old_zoom
            self._offset = center - (center - self._offset) * ratio
            self._zoom = clamped_zoom
            self.zoom_changed.emit(int(self._zoom * 100))
            self.update()

    def _get_image_rect(self) -> QRectF:
        """คำนวณตำแหน่งและขนาดภาพบนหน้าจอตามค่า zoom และ offset ปัจจุบัน"""
        ref_w, ref_h = self._get_ref_size()
        if ref_w <= 0 or ref_h <= 0:
            return QRectF()
        return QRectF(
            self._offset.x(),
            self._offset.y(),
            ref_w * self._zoom,
            ref_h * self._zoom,
        )

    # ──────────────────────────────────────
    # Paint Event
    # ──────────────────────────────────────
    def paintEvent(self, event: QPaintEvent) -> None:
        """วาดการแสดงผลภาพตามโหมดที่เลือก"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))

        # หากยังไม่มีภาพ
        if self._before_qimg is None or self._before_qimg.isNull():
            painter.setPen(QColor("#666666"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "ยังไม่มีรูปภาพ\nลากและวางรูปภาพที่นี่ หรือกด Ctrl+O เพื่อเลือกไฟล์",
            )
            painter.end()
            return

        # ── กรณีที่ 1: โหมด Split Slider (เลื่อนเปรียบเทียบ) ──
        if self._mode == self.MODE_COMPARE_SLIDER and self._after_qimg is not None:
            self._paint_split_slider(painter)

        # ── กรณีที่ 2: โหมด Side-by-Side (คู่กัน ซ้าย-ขวา) ──
        elif self._mode == self.MODE_SIDE_BY_SIDE and self._after_qimg is not None:
            self._paint_side_by_side(painter)

        # ── กรณีที่ 3: โหมด Single Image (ภาพเดี่ยว) ──
        else:
            self._paint_single(painter)

        # วาด Pixel Inspector HUD ถ้ามีข้อมูล
        self._paint_pixel_hud(painter)

        painter.end()

    def _paint_split_slider(self, painter: QPainter) -> None:
        """วาดภาพในโหมดสไลเดอร์เปรียบเทียบซ้อนทับ"""
        rect = self._get_image_rect()
        split_x = int(self.width() * self._slider_pos)

        # วาดภาพ "หลัง" (After) ฝั่งซ้าย
        painter.save()
        painter.setClipRect(0, 0, split_x, self.height())
        painter.drawImage(rect, self._after_qimg)
        painter.restore()

        # วาดภาพ "ก่อน" (Before) ฝั่งขวา (ขยายให้ตรงกับขนาดและตำแหน่งของ rect เดียวกัน!)
        painter.save()
        painter.setClipRect(split_x, 0, self.width() - split_x, self.height())
        painter.drawImage(rect, self._before_qimg)
        painter.restore()

        # ป้ายกำกับ "หลัง (AI/ขยาย)" ฝั่งซ้าย
        self._draw_badge(painter, 16, 16, "หลัง (ขยายแล้ว)", QColor(46, 125, 50, 220))

        # ป้ายกำกับ "ก่อน" ฝั่งขวา
        self._draw_badge(painter, self.width() - 96, 16, "ก่อน (ต้นฉบับ)", QColor(25, 118, 210, 220))

        # วาดเส้นแบ่งกึ่งกลางสไลเดอร์
        painter.setPen(QPen(QColor(255, 255, 255, 230), self._slider_width))
        painter.drawLine(split_x, 0, split_x, self.height())

        # วาดปุ่มจับเลื่อนวงกลม (Slider Handle)
        handle_y = self.height() // 2
        radius = 18
        painter.setPen(QPen(QColor(255, 255, 255, 240), 2))
        painter.setBrush(QBrush(QColor(33, 150, 243, 230)))
        painter.drawEllipse(QPoint(split_x, handle_y), radius, radius)

        # วาดลูกศร ◀ ▶
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(
            split_x - radius, handle_y - radius,
            radius * 2, radius * 2,
            Qt.AlignmentFlag.AlignCenter, "◀ ▶"
        )

    def _paint_side_by_side(self, painter: QPainter) -> None:
        """วาดภาพในโหมดแสดงคู่กัน ซ้าย: ก่อน | ขวา: หลัง"""
        half_w = self.width() / 2
        ref_w, ref_h = self._get_ref_size()

        # ฝั่งซ้าย: ก่อน
        left_rect = QRectF(
            self._offset.x(),
            self._offset.y(),
            ref_w * self._zoom,
            ref_h * self._zoom,
        )
        painter.save()
        painter.setClipRect(0, 0, int(half_w), self.height())
        painter.drawImage(left_rect, self._before_qimg)
        self._draw_badge(painter, 16, 16, "ต้นฉบับ (ก่อน)", QColor(25, 118, 210, 220))
        painter.restore()

        # ฝั่งขวา: หลัง
        right_rect = QRectF(
            self._offset.x() + half_w,
            self._offset.y(),
            ref_w * self._zoom,
            ref_h * self._zoom,
        )
        painter.save()
        painter.setClipRect(int(half_w), 0, int(half_w), self.height())
        painter.drawImage(right_rect, self._after_qimg)
        self._draw_badge(painter, int(half_w) + 16, 16, "ผลลัพธ์ (หลัง)", QColor(46, 125, 50, 220))
        painter.restore()

        # เส้นคั่นกลาง
        painter.setPen(QPen(QColor("#444444"), 2))
        painter.drawLine(int(half_w), 0, int(half_w), self.height())

    def _paint_single(self, painter: QPainter) -> None:
        """วาดภาพในโหมดภาพเดี่ยว"""
        target_img = self._after_qimg if self._after_qimg is not None else self._before_qimg
        rect = self._get_image_rect()
        painter.drawImage(rect, target_img)

        # ป้ายกำกับ
        label_text = "ผลลัพธ์" if self._after_qimg is not None else "ต้นฉบับ"
        bg_color = QColor(46, 125, 50, 220) if self._after_qimg is not None else QColor(25, 118, 210, 220)
        self._draw_badge(painter, 16, 16, label_text, bg_color)

    def _draw_badge(self, painter: QPainter, x: int, y: int, text: str, color: QColor) -> None:
        """วาดกล่องข้อความป้ายกำกับโปร่งแสง"""
        painter.save()
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(text) + 16
        text_h = 24

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(x, y, text_w, text_h, 4, 4)

        painter.setPen(QColor("#ffffff"))
        painter.drawText(x, y, text_w, text_h, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def _paint_pixel_hud(self, painter: QPainter) -> None:
        """วาด Pixel Inspector HUD ที่มุมล่างซ้าย"""
        if self._hover_pixel is None:
            return

        img_x, img_y, r, g, b = self._hover_pixel
        hud_text = f"X: {img_x} | Y: {img_y} | RGB: ({r}, {g}, {b}) | ซูม: {int(self._zoom * 100)}%"

        painter.save()
        painter.setFont(QFont("Consolas", 9))
        metrics = painter.fontMetrics()
        hud_w = metrics.horizontalAdvance(hud_text) + 36
        hud_h = 26
        hud_x = 16
        hud_y = self.height() - hud_h - 14

        # พื้นหลัง HUD
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.setBrush(QBrush(QColor(26, 26, 26, 210)))
        painter.drawRoundedRect(hud_x, hud_y, hud_w, hud_h, 4, 4)

        # ช่องสี่เหลี่ยมสีตัวอย่าง
        sample_size = 12
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.setBrush(QBrush(QColor(r, g, b)))
        painter.drawRect(hud_x + 8, hud_y + (hud_h - sample_size) // 2, sample_size, sample_size)

        # ข้อความพิกัดและค่าสี
        painter.setPen(QColor("#cccccc"))
        painter.drawText(
            hud_x + 26, hud_y,
            hud_w - 26, hud_h,
            Qt.AlignmentFlag.AlignVCenter, hud_text
        )
        painter.restore()

    # ──────────────────────────────────────
    # Events
    # ──────────────────────────────────────
    def wheelEvent(self, event: QWheelEvent) -> None:
        """ซูมเข้า/ออกโดยอิงตำแหน่งเมาส์"""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        factor = 1.18 if delta > 0 else (1.0 / 1.18)
        mouse_pos = QPointF(event.position())
        self._set_zoom(self._zoom * factor, center_point=mouse_pos)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """กดปุ่มคีย์บอร์ด (Spacebar เพื่อ Pan)"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """ปล่อยปุ่มคีย์บอร์ด"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """คลิกเมาส์เพื่อจับ slider หรือเริ่ม pan"""
        if event.button() == Qt.MouseButton.LeftButton:
            # ตรวจสอบการคลิกที่ตัวเลื่อนเปรียบเทียบในโหมด Split Slider
            if self._mode == self.MODE_COMPARE_SLIDER and self._after_qimg is not None:
                split_x = int(self.width() * self._slider_pos)
                if abs(event.position().x() - split_x) <= 24:
                    self._slider_dragging = True
                    self.setCursor(Qt.CursorShape.SplitHCursor)
                    return

            # เริ่มต้น Pan ภาพ
            self._panning = True
            self._pan_start = event.position().toPoint()
            self._pan_offset_start = QPointF(self._offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """เมื่อเลื่อนเมาส์"""
        pos = event.position().toPoint()
        self._cursor_pos = pos

        # 1. ลากสไลเดอร์แบ่งภาพ
        if self._slider_dragging:
            self._slider_pos = max(0.01, min(0.99, pos.x() / self.width()))
            self.update()
            return

        # 2. ลาก Pan ภาพ
        if self._panning:
            delta = pos - self._pan_start
            self._offset = QPointF(
                self._pan_offset_start.x() + delta.x(),
                self._pan_offset_start.y() + delta.y(),
            )
            self.update()
            return

        # 3. อัพเดท Cursor และ Pixel Inspector HUD
        if self._mode == self.MODE_COMPARE_SLIDER and self._after_qimg is not None:
            split_x = int(self.width() * self._slider_pos)
            if abs(pos.x() - split_x) <= 24:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            elif self._space_pressed:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        elif self._space_pressed:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        self._update_hover_pixel(pos)
        self.update()

    def _update_hover_pixel(self, widget_pos: QPoint) -> None:
        """คำนวณพิกัดพิกเซลในภาพและดึงค่าสี RGB"""
        rect = self._get_image_rect()
        if not rect.contains(QPointF(widget_pos)):
            self._hover_pixel = None
            return

        # คำนวณพิกัดในภาพ
        ref_w, ref_h = self._get_ref_size()
        rel_x = (widget_pos.x() - rect.left()) / rect.width()
        rel_y = (widget_pos.y() - rect.top()) / rect.height()

        px_x = int(rel_x * ref_w)
        px_y = int(rel_y * ref_h)
        px_x = max(0, min(ref_w - 1, px_x))
        px_y = max(0, min(ref_h - 1, px_y))

        # ดึงสีจาก active image
        active_img = self._after_img if self._after_img is not None else self._before_img
        if active_img is not None and px_y < active_img.shape[0] and px_x < active_img.shape[1]:
            pixel = active_img[px_y, px_x]
            if len(pixel) >= 3:
                b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
                self._hover_pixel = (px_x, px_y, r, g, b)
                self.pixel_hovered.emit(px_x, px_y, r, g, b)
                return

        self._hover_pixel = None

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """ปล่อยการคลิกเมาส์"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._slider_dragging = False
            self._panning = False
            if self._space_pressed:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """ดับเบิลคลิกเพื่อซูมพอดีหน้าต่าง (Fit to Screen)"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.zoom_fit()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """ปรับขนาดอัตโนมัติเมื่อขนาดหน้าต่างเปลี่ยน"""
        super().resizeEvent(event)
        if self._before_qimg is not None:
            self._fit_image()
            self.update()
