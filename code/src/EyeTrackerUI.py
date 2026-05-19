# EyeTrackerUI.py 
import sys
import cv2
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional
import random
import math
import webbrowser
import os
import subprocess

# Импорт ядра трекера
from EyeTracker import (
    AdvancedGazeTracker,
    MovementBasedCryptoGenerator,
    MEDIAPIPE_AVAILABLE
)

# Импорт модуля голосовой энтропии
from VoiceEntropy import (
    VoiceEntropyCollector,
    generate_combined_key,
    SOUNDDEVICE_AVAILABLE,
)

# ─── Константы минимального пула энтропии ────────────────────────────────────
# Получены калибровкой через ea_non_iid (SP800-90B) на реальных данных.
# Результат: min_entropy = 6.637 бит/байт (файл entropy_pool_20260517.bin)
#
# Формула: MIN_BYTES = ceil(key_bits / H_min) * SAFETY_FACTOR
#   H_MIN_BITS_PER_BYTE = 6.637   — консервативная оценка из SP800-90B
#   SAFETY_FACTOR       = 3       — запас надёжности (NIST рекомендует ≥1.5)
#
#   128 бит: ceil(128 / 6.637) * 3 = ceil(19.28) * 3 = 20 * 3 = 60 байт
#   256 бит: ceil(256 / 6.637) * 3 = ceil(38.57) * 3 = 39 * 3 = 117 байт

H_MIN_BITS_PER_BYTE  = 6.637   # min-entropy из ea_non_iid, бит/байт
SAFETY_FACTOR        = 3       # коэффициент запаса

MIN_POOL_BYTES_128   = 60      # минимум байт пула для 128-бит ключа
MIN_POOL_BYTES_256   = 117     # минимум байт пула для 256-бит ключа

# Импорт PyQt5
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtMultimedia import *
    from PyQt5.QtMultimediaWidgets import *
except ImportError:
    print("❌ PyQt5 не установлен. Установите: pip install PyQt5")
    sys.exit(1)


class DarkTheme:
    """Класс для применения темной темы"""
    
    @staticmethod
    def apply(app: QApplication):
        app.setStyle("Fusion")
        
        dark_palette = QPalette()
        
        # Основные цвета темной темы
        dark_palette.setColor(QPalette.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.Base, QColor(20, 20, 20))
        dark_palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.Text, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 100, 100))
        dark_palette.setColor(QPalette.Link, QColor(66, 165, 245))
        dark_palette.setColor(QPalette.Highlight, QColor(66, 165, 245))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        
        # Цвета для disabled элементов
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(128, 128, 128))
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
        
        app.setPalette(dark_palette)


class LoadingOverlay(QWidget):
    """Оверлей с индикатором загрузки"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.animation_angle = 0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30)
        self.hide()
        
    def update_animation(self):
        self.animation_angle = (self.animation_angle + 10) % 360
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.fillRect(self.rect(), QColor(20, 20, 20, 220))
        
        center_x = self.width() // 2
        center_y = self.height() // 2
        spinner_size = min(60, self.width() // 4, self.height() // 4)
        
        painter.translate(center_x, center_y)
        painter.rotate(self.animation_angle)
        
        for i in range(8):
            painter.rotate(45)
            opacity = 255 - (i * 25)
            color = QColor(76, 175, 80, opacity)
            
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            
            rect = QRect(-spinner_size // 6, -spinner_size // 2, 
                        spinner_size // 3, spinner_size // 5)
            painter.drawRoundedRect(rect, 3, 3)
        
        painter.resetTransform()
        
        painter.setPen(QColor(220, 220, 220))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, "Загрузка камеры...")
        
        painter.setPen(QColor(160, 160, 160))
        painter.setFont(QFont("Segoe UI", 9))
        text_rect = QRect(0, center_y + spinner_size, self.width(), 30)
        painter.drawText(text_rect, Qt.AlignCenter, "Пожалуйста, подождите")
        
    def showEvent(self, event):
        self.timer.start()
        super().showEvent(event)
        
    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)


class CameraPlaceholder(QWidget):
    """Виджет-заглушка с пиктограммой камеры"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        self.pulse_alpha = 80
        self.pulse_direction = 1
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.update_pulse)
        
    def update_pulse(self):
        self.pulse_alpha += self.pulse_direction * 2
        if self.pulse_alpha >= 120:
            self.pulse_alpha = 120
            self.pulse_direction = -1
        elif self.pulse_alpha <= 60:
            self.pulse_alpha = 60
            self.pulse_direction = 1
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        
        center_x = self.width() // 2
        center_y = self.height() // 2
        icon_size = min(100, self.width() // 3, self.height() // 3)
        
        painter.setBrush(QBrush(QColor(76, 175, 80, self.pulse_alpha)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPoint(center_x, center_y), icon_size // 2 + 10, icon_size // 2 + 10)
        
        painter.setPen(QPen(QColor(120, 120, 120), 3))
        painter.setBrush(QBrush(QColor(80, 80, 80, 100)))
        
        camera_rect = QRect(center_x - icon_size // 2, center_y - icon_size // 3, 
                           icon_size, icon_size * 2 // 3)
        painter.drawRoundedRect(camera_rect, 8, 8)
        
        lens_size = icon_size // 3
        lens_rect = QRect(center_x - lens_size // 2, center_y - lens_size // 2, 
                         lens_size, lens_size)
        
        gradient = QRadialGradient(center_x - lens_size // 4, center_y - lens_size // 4, lens_size)
        gradient.setColorAt(0, QColor(76, 175, 80))
        gradient.setColorAt(1, QColor(40, 100, 40))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(160, 160, 160), 2))
        painter.drawEllipse(lens_rect)
        
        painter.setBrush(QBrush(QColor(255, 255, 255, 100)))
        painter.setPen(Qt.NoPen)
        blink_rect = QRect(center_x - lens_size // 6, center_y - lens_size // 3, 
                          lens_size // 3, lens_size // 4)
        painter.drawEllipse(blink_rect)
        
        flash_size = icon_size // 6
        flash_rect = QRect(center_x + icon_size // 3, center_y - icon_size // 4, 
                          flash_size, flash_size)
        painter.setBrush(QBrush(QColor(160, 160, 160, 100)))
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.drawRoundedRect(flash_rect, 3, 3)
        
        painter.setPen(QColor(160, 160, 160))
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        text_rect = QRect(0, center_y + icon_size // 2 + 20, self.width(), 30)
        painter.drawText(text_rect, Qt.AlignCenter, "Камера остановлена")
        
        painter.setPen(QColor(100, 100, 100))
        painter.setFont(QFont("Segoe UI", 9))
        subtext_rect = QRect(0, center_y + icon_size // 2 + 45, self.width(), 20)
        painter.drawText(subtext_rect, Qt.AlignCenter, "Нажмите ▶ для запуска")
        
    def showEvent(self, event):
        self.pulse_timer.start(50)
        super().showEvent(event)
        
    def hideEvent(self, event):
        self.pulse_timer.stop()
        super().hideEvent(event)


class VideoWidget(QWidget):
    """Виджет для отображения видео с индикатором загрузки и заглушкой"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        video_label_widget = QLabel("Видео")
        video_label_widget.setStyleSheet("font-size: 12px; color: #aaa; padding: 5px;")
        layout.addWidget(video_label_widget)
        
        self.video_container = QWidget()
        self.video_container.setMinimumSize(400, 300)
        self.video_container.setStyleSheet("""
            QWidget {
                border-radius: 8px;
                background-color: #1a1a1a;
            }
        """)
        
        container_layout = QVBoxLayout(self.video_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(400, 300)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            border-radius: 8px;
            background-color: #1a1a1a;
        """)
        container_layout.addWidget(self.video_label)
        
        self.loading_overlay = LoadingOverlay(self.video_container)
        self.loading_overlay.hide()
        
        self.camera_placeholder = CameraPlaceholder(self.video_container)
        self.camera_placeholder.hide()
        
        layout.addWidget(self.video_container)
        self.setLayout(layout)
        
        self.first_frame_received = False
        self.is_camera_active = False
        
    def show_loading(self):
        self.first_frame_received = False
        self.is_camera_active = True
        self.camera_placeholder.hide()
        self.loading_overlay.setGeometry(self.video_container.rect())
        self.loading_overlay.show()
        self.video_label.clear()
        
    def hide_loading(self):
        self.loading_overlay.hide()
        self.first_frame_received = True
        
    def show_placeholder(self):
        self.is_camera_active = False
        self.loading_overlay.hide()
        self.camera_placeholder.setGeometry(self.video_container.rect())
        self.camera_placeholder.show()
        self.video_label.clear()
        
    def hide_placeholder(self):
        self.camera_placeholder.hide()
        
    def setPixmap(self, pixmap):
        if self.is_camera_active:
            if not self.first_frame_received and pixmap and not pixmap.isNull():
                self.hide_loading()
            
            if self.first_frame_received:
                self.camera_placeholder.hide()
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
        
    def clear(self):
        self.video_label.clear()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.loading_overlay.setGeometry(self.video_container.rect())
        self.camera_placeholder.setGeometry(self.video_container.rect())


class AudioVisualizerWidget(QWidget):
    """
    Виджет визуализации звука — подключён к VoiceEntropyCollector.
    Отображает реальные уровни амплитуды из микрофона.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        self.audio_levels = deque(maxlen=40)
        self.is_recording = False

        # Реальный коллектор голосовой энтропии
        self.voice_collector = VoiceEntropyCollector()

        # Таймер обновления визуализации (50 мс = 20 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self._pull_real_levels)

        self.setStyleSheet("""
            AudioVisualizerWidget {
                background-color: #1a1a1a;
                border-radius: 8px;
            }
        """)

    # ── Получение реальных уровней из коллектора ──────────────────────────

    def _pull_real_levels(self):
        """Вытягивает актуальные уровни из VoiceEntropyCollector."""
        stats = self.voice_collector.get_statistics()
        levels = stats.get("audio_levels", [])

        if levels:
            for lvl in levels[-4:]:          # добавляем последние 4 уровня
                self.audio_levels.append(lvl)
        else:
            # Тишина — добавляем очень маленький уровень
            self.audio_levels.append(0.02)

        self.update()

    # ── Рисование ─────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#1a1a1a"))
        painter.drawRoundedRect(self.rect(), 8, 8)

        # Заголовок
        painter.setPen(QColor("#e0e0e0"))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.drawText(15, 30, "🎤 Аудио")

        # Индикатор активности
        status_color = QColor("#4CAF50") if self.is_recording else QColor("#666666")
        painter.setBrush(QBrush(status_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.width() - 30, 15, 8, 8)

        painter.setPen(QColor("#aaa"))
        painter.setFont(QFont("Segoe UI", 9))
        status_text = "Запись" if self.is_recording else "Ожидание"
        painter.drawText(self.width() - 50, 28, status_text)

        # Полоски эквалайзера
        if self.audio_levels:
            bar_width = 3
            spacing = 2
            total_width = len(self.audio_levels) * (bar_width + spacing)
            start_x = (self.width() - total_width) // 2

            for i, level in enumerate(self.audio_levels):
                x = start_x + i * (bar_width + spacing)
                height = int(level * 120)

                if level > 0.7:
                    color = QColor("#c0c0c0")
                elif level > 0.4:
                    color = QColor("#909090")
                else:
                    color = QColor("#606060")

                painter.setBrush(QBrush(color))
                painter.setPen(Qt.NoPen)
                rect = QRect(x, self.height() // 2 - height // 2, bar_width, max(2, height))
                painter.drawRoundedRect(rect, 1.5, 1.5)

        # Центральная линия
        painter.setPen(QPen(QColor("#404040"), 0.5))
        painter.drawLine(15, self.height() // 2, self.width() - 15, self.height() // 2)

        # Подпись статистики
        painter.setPen(QColor("#888"))
        painter.setFont(QFont("Segoe UI", 8))
        if self.audio_levels:
            avg_lvl = sum(self.audio_levels) / len(self.audio_levels)
            peak_lvl = max(self.audio_levels)
            painter.drawText(15, self.height() - 20, f"Средний: {avg_lvl:.2f}")
            painter.drawText(15, self.height() - 8,  f"Пиковый: {peak_lvl:.2f}")

        # Подпись источника
        painter.setPen(QColor("#666"))
        painter.setFont(QFont("Segoe UI", 7))
        src = "Реальный микрофон" if SOUNDDEVICE_AVAILABLE else "sounddevice не установлен"
        painter.drawText(self.width() - 200, self.height() - 8, src)

    # ── Управление ────────────────────────────────────────────────────────

    def start_recording(self):
        ok = self.voice_collector.start()
        if ok:
            self.is_recording = True
            self.timer.start(50)
            print("🎤 Микрофон запущен, сбор голосовой энтропии активен")
        else:
            self.is_recording = False
            print("⚠ Микрофон недоступен — голосовая энтропия не будет использована")

    def stop_recording(self):
        self.is_recording = False
        self.voice_collector.stop()
        self.timer.stop()
        self.audio_levels.clear()

    def reset(self):
        self.voice_collector.reset()
        self.audio_levels.clear()


class CameraThread(QThread):
    """Поток для захвата и обработки видео"""
    frame_ready = pyqtSignal(object, object)
    stats_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    camera_started = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.tracker = AdvancedGazeTracker()
        self.crypto_generator = MovementBasedCryptoGenerator(self.tracker)
        self.running = False
        self.cap = None
        
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.error_occurred.emit("Не удалось открыть камеру")
            return
        
        self.camera_started.emit()
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                processed_frame, gaze_data = self.tracker.process_frame(frame)
                self.frame_ready.emit(processed_frame, gaze_data)
                
                if self.tracker.frame_count % 10 == 0:
                    stats = self.tracker.get_movement_statistics()
                    self.stats_updated.emit(stats)
            
            self.msleep(10)
            
        if self.cap:
            self.cap.release()
    
    def stop(self):
        self.running = False
        self.wait()
    
    def generate_key(self, bits: int = 256):
        return self.crypto_generator.generate_secure_bits(bits)
    
    def reset_data(self):
        self.tracker.movement_events.clear()
        self.tracker.gaze_history.clear()
        self.tracker.eye_state_history.clear()
        self.tracker.consecutive_still_frames = 0
        self.tracker.last_movement_time = 0
        self.tracker.last_valid_gaze = None
        self.tracker.frame_count = 0


class StarWidget(QWidget):
    """Виджет с одной минималистичной звездой"""
    
    def __init__(self, parent=None, star_id=0, x=100, y=100):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.star_id = star_id
        self.size = random.randint(15, 35)
        self.rotation = random.uniform(0, 360)
        self.x = x
        self.y = y
        self.color = QColor(255, 255, 0)
        self.lifetime = random.randint(800, 2000)
        self.creation_time = QDateTime.currentMSecsSinceEpoch()
        self.is_visible = True
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_lifetime)
        self.timer.start(50)
        
        self.resize(self.size + 20, self.size + 20)
        self.move(int(self.x), int(self.y))
        
    def check_lifetime(self):
        current_time = QDateTime.currentMSecsSinceEpoch()
        elapsed = current_time - self.creation_time
        
        if elapsed > self.lifetime:
            self.stop_animation()
            if self.parent():
                self.parent().star_died.emit(self.star_id)
        else:
            self.update()
    
    def paintEvent(self, event):
        if not self.is_visible:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.save()
        painter.translate(self.size // 2 + 10, self.size // 2 + 10)
        painter.rotate(self.rotation)
        
        scale = self.size / 30
        painter.scale(scale, scale)
        
        star_color = QColor(255, 255, 0)
        glow_color = QColor(255, 255, 0, 80)
        painter.setBrush(QBrush(glow_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(-12, -12, 24, 24)
        
        painter.setBrush(QBrush(star_color))
        painter.setPen(QPen(QColor(255, 200, 0), 1.5))
        
        path = QPainterPath()
        points = [(0, -15), (10, 0), (0, 15), (-10, 0)]
        
        path.moveTo(points[0][0], points[0][1])
        for i in range(len(points)):
            current = points[i]
            next_point = points[(i + 1) % len(points)]
            cx1 = current[0] * 0.5
            cy1 = current[1] * 0.5
            cx2 = next_point[0] * 0.5
            cy2 = next_point[1] * 0.5
            path.cubicTo(cx1, cy1, cx2, cy2, next_point[0], next_point[1])
        
        painter.drawPath(path)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(-2, -2, 4, 4)
        painter.restore()
    
    def start_animation(self):
        self.show()
        self.timer.start()
    
    def stop_animation(self):
        self.timer.stop()
        self.hide()


class StarField(QWidget):
    """Управление полем звезд"""
    star_died = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.parent_widget = parent
        
        self.stars = {}
        self.next_star_id = 0
        self.min_stars = 10
        self.max_stars = 25
        self.spawn_timer = QTimer()
        self.spawn_timer.timeout.connect(self.spawn_star)
        self.is_active = False
        
    def get_random_position(self):
        if self.parent_widget:
            rect = self.parent_widget.geometry()
            x = random.randint(50, rect.width() - 100)
            y = random.randint(50, rect.height() - 100)
            return x, y
        else:
            screen = QApplication.primaryScreen().geometry()
            return random.randint(50, screen.width() - 100), random.randint(50, screen.height() - 100)
    
    def spawn_star(self):
        if not self.is_active:
            return
            
        if len(self.stars) < self.max_stars:
            star_id = self.next_star_id
            self.next_star_id += 1
            x, y = self.get_random_position()
            star = StarWidget(self, star_id, x, y)
            self.stars[star_id] = star
            star.start_animation()
    
    def remove_star(self, star_id):
        if star_id in self.stars:
            star = self.stars[star_id]
            star.stop_animation()
            star.deleteLater()
            del self.stars[star_id]
    
    def start_field(self):
        self.is_active = True
        for star_id in list(self.stars.keys()):
            self.remove_star(star_id)
        
        initial_count = random.randint(self.min_stars, self.max_stars)
        for _ in range(initial_count):
            self.spawn_star()
        
        self.spawn_timer.start(random.randint(300, 700))
    
    def stop_field(self):
        self.is_active = False
        self.spawn_timer.stop()
        for star_id in list(self.stars.keys()):
            self.remove_star(star_id)
    
    def handle_star_died(self, star_id):
        self.remove_star(star_id)
        if self.is_active:
            self.spawn_star()


class InstructionWidget(QWidget):
    """Виджет с инструкцией по работе с программой"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📚 Инструкция")
        self.setGeometry(200, 200, 600, 500)
        self.setStyleSheet("background-color: #2a2a2a;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("📚 Руководство по использованию")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #e0e0e0; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d0d0d0;
                border: 1px solid #404040;
                border-radius: 8px;
                font-size: 13px;
                padding: 15px;
            }
        """)
        
        instructions = """
<h3 style="color: #4CAF50;">🎯 Общая информация</h3>
<p>EyeTracker генерирует криптографические ключи на основе движений глаз и голоса.</p>

<h3 style="color: #2196F3;">🚀 Начало работы</h3>
<p>1. Нажмите <b>▶ Начать трекинг</b><br>
2. Следите за стимулами на экране<br>
3. Накопите минимум 20 движений</p>

<h3 style="color: #FF9800;">🎮 Элементы управления</h3>
<p>• <b>YouTube/Browser</b> - источники стимулов<br>
• <b>Случайные триггеры</b> - звезды на экране<br>
• <b>Сбросить данные</b> - очистка статистики</p>

<h3 style="color: #9C27B0;">🔐 Генерация ключей</h3>
<p>• <b>128 бит</b> - базовый уровень<br>
• <b>256 бит</b> - повышенная безопасность</p>
"""
        
        text.setHtml(instructions)
        layout.addWidget(text)
        
        close_btn = QPushButton("Закрыть")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)


class ModernToggle(QWidget):
    """Современный переключатель"""
    toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None, text="", initial_state=False):
        super().__init__(parent)
        self.state = initial_state
        self.text = text
        self.setFixedSize(200, 40)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        bg_color = QColor("#4CAF50") if self.state else QColor("#505050")
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 20, 20)
        
        knob_x = self.width() - 38 if self.state else 2
        painter.setBrush(QBrush(QColor("#e0e0e0")))
        painter.drawEllipse(int(knob_x), 2, 36, 36)
        
        painter.setPen(QColor("#d0d0d0"))
        painter.setFont(QFont("Segoe UI", 10))
        text_x = 50 if not self.state else 10
        painter.drawText(QRect(text_x, 0, 150, 40), Qt.AlignVCenter, self.text)
    
    def mousePressEvent(self, event):
        self.state = not self.state
        self.toggled.emit(self.state)
        self.update()


class TripleToggle(QWidget):
    """
    Тумблер с тремя фиксированными положениями.
    Положения: 0 = левое, 1 = среднее, 2 = правое.
    Рукоятка плавно скользит между позициями при клике.
    """
    changed = pyqtSignal(int)   # 0 | 1 | 2

    LABELS   = ["👁 Взгляд", "👁+🎤 Оба", "🎤 Голос"]
    COLORS   = ["#64B5F6",   "#4CAF50",   "#FF9800"]
    SOURCES  = ["eye",       "both",      "voice"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pos   = 1          # текущее положение (0/1/2), по умолчанию «Оба»
        self._anim  = 0.0        # анимированная позиция рукоятки 0.0–2.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(280, 52)
        self.setCursor(Qt.PointingHandCursor)

    # ── Анимация ──────────────────────────────────────────────────────────
    def _tick(self):
        target = float(self._pos)
        diff   = target - self._anim
        if abs(diff) < 0.02:
            self._anim = target
            self._timer.stop()
        else:
            self._anim += diff * 0.25
        self.update()

    # ── Отрисовка ─────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W, H   = self.width(), self.height()
        R      = 14          # радиус скругления трека
        PAD    = 4           # отступ рукоятки от края трека
        SLOT_W = (W - PAD * 2) / 3   # ширина одного слота

        # ── Трек ─────────────────────────────────────────────────────────
        track_color = QColor(self.COLORS[self._pos])
        track_color.setAlpha(60)
        p.setBrush(QBrush(track_color))
        p.setPen(QPen(QColor(self.COLORS[self._pos]), 1.5))
        p.drawRoundedRect(0, 0, W, H, R, R)

        # ── Метки позиций ────────────────────────────────────────────────
        for i, label in enumerate(self.LABELS):
            cx = int(PAD + SLOT_W * i + SLOT_W / 2)
            active = (i == self._pos)
            p.setPen(QColor("#e0e0e0") if active else QColor("#707070"))
            p.setFont(QFont("Segoe UI", 9, QFont.Bold if active else QFont.Normal))
            p.drawText(QRect(int(PAD + SLOT_W * i), 0, int(SLOT_W), H),
                       Qt.AlignCenter, label)

        # ── Рукоятка (скользит по _anim) ─────────────────────────────────
        knob_x = int(PAD + self._anim * SLOT_W)
        knob_w = int(SLOT_W - PAD)
        knob_h = H - PAD * 2

        knob_color = QColor(self.COLORS[self._pos])
        p.setBrush(QBrush(knob_color))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(knob_x, PAD, knob_w, knob_h, R - 2, R - 2)

        # Текст на рукоятке
        p.setPen(QColor("#ffffff"))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(QRect(knob_x, PAD, knob_w, knob_h),
                   Qt.AlignCenter, self.LABELS[self._pos])

    # ── Клик — определяем в какой слот попал курсор ───────────────────────
    def mousePressEvent(self, event):
        if not self.isEnabled():
            return
        SLOT_W = self.width() / 3
        clicked = int(event.x() // SLOT_W)
        clicked = max(0, min(2, clicked))
        if clicked != self._pos:
            self._pos = clicked
            self._timer.start(16)   # ~60 FPS
            self.changed.emit(self._pos)
            self.update()

    # ── Публичный API ─────────────────────────────────────────────────────
    def source(self) -> str:
        return self.SOURCES[self._pos]

    def set_position(self, pos: int):
        self._pos  = max(0, min(2, pos))
        self._anim = float(self._pos)
        self.update()


class ModernCard(QFrame):
    """Современная карточка"""
    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.setStyleSheet("""
            ModernCard {
                background-color: #2a2a2a;
                border-radius: 12px;
                border: 1px solid #404040;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                font-size: 16px;
                font-weight: 600;
                color: #e0e0e0;
                padding-bottom: 5px;
            """)
            layout.addWidget(title_label)
        
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)
        self.setLayout(layout)
    
    def addWidget(self, widget):
        if isinstance(widget, QLayout):
            self.content_layout.addLayout(widget)
        else:
            self.content_layout.addWidget(widget)


class ModernButton(QPushButton):
    """Современная кнопка"""
    def __init__(self, text, primary=False, accent=False):
        super().__init__(text)
        self.setFixedHeight(44)
        self.setCursor(Qt.PointingHandCursor)
        
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
                QPushButton:disabled {
                    background-color: #505050;
                    color: #888;
                }
            """)
        elif accent:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #1e88e5;
                }
                QPushButton:pressed {
                    background-color: #1976d2;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #c0c0c0;
                    border: 1px solid #505050;
                    border-radius: 8px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #3a3a3a;
                    border-color: #606060;
                }
            """)


class IconButton(QPushButton):
    """Кнопка с иконкой"""
    def __init__(self, icon, text="", color="#c0c0c0"):
        super().__init__(f"{icon} {text}" if text else icon)
        if not text:
            self.setFixedSize(40, 40)
        else:
            self.setFixedHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {color};
                border: none;
                border-radius: 8px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: #3a3a3a;
            }}
        """)


class StatisticsWidget(QWidget):
    """Виджет для отображения статистики взгляда"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(0, 0, 0, 0)
        
        stats_card = ModernCard(title="📊 Статистика взгляда")
        
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(15)
        
        self.metrics = {}
        metrics_data = [
            ("total_frames", "Кадры", "0", "#2196F3"),
            ("movement_count", "Движения", "0", "#4CAF50"),
            ("movement_rate", "Частота", "0%", "#FF9800"),
            ("data_quality", "Качество", "0%", "#9C27B0"),
            ("avg_magnitude", "Амплитуда", "0 px", "#F44336"),
        ]
        
        for i, (key, label, default, color) in enumerate(metrics_data):
            metric_widget = QWidget()
            metric_layout = QVBoxLayout()
            metric_layout.setSpacing(5)
            
            value_label = QLabel(default)
            value_label.setStyleSheet(f"""
                font-size: 28px;
                font-weight: 700;
                color: {color};
            """)
            value_label.setAlignment(Qt.AlignCenter)
            
            name_label = QLabel(label)
            name_label.setStyleSheet("font-size: 12px; color: #aaa;")
            name_label.setAlignment(Qt.AlignCenter)
            
            metric_layout.addWidget(value_label)
            metric_layout.addWidget(name_label)
            metric_widget.setLayout(metric_layout)
            
            row = i // 2
            col = i % 2
            metrics_grid.addWidget(metric_widget, row, col)
            self.metrics[key] = value_label
        
        stats_card.addWidget(metrics_grid)
        
        entropy_widget = QWidget()
        entropy_layout = QVBoxLayout()
        
        entropy_header = QHBoxLayout()
        entropy_header.addWidget(QLabel("🔐 Качество энтропии"))
        self.entropy_value = QLabel("0%")
        self.entropy_value.setStyleSheet("font-weight: 600; color: #aaa;")
        entropy_header.addWidget(self.entropy_value)
        entropy_layout.addLayout(entropy_header)
        
        self.entropy_bar = QProgressBar()
        self.entropy_bar.setMaximum(100)
        self.entropy_bar.setTextVisible(False)
        self.entropy_bar.setFixedHeight(8)
        self.entropy_bar.setStyleSheet("""
            QProgressBar {
                background-color: #3a3a3a;
                border-radius: 4px;
                border: none;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #8BC34A);
                border-radius: 4px;
            }
        """)
        entropy_layout.addWidget(self.entropy_bar)
        
        self.entropy_label = QLabel("Ожидание данных...")
        self.entropy_label.setStyleSheet("font-size: 12px; color: #888;")
        entropy_layout.addWidget(self.entropy_label)
        
        entropy_widget.setLayout(entropy_layout)
        stats_card.addWidget(entropy_widget)
        
        layout.addWidget(stats_card)
        self.setLayout(layout)
    
    def update_stats(self, stats: dict):
        self.metrics["total_frames"].setText(str(stats.get('total_frames', 0)))
        self.metrics["movement_count"].setText(str(stats.get('movement_count', 0)))
        self.metrics["movement_rate"].setText(f"{stats.get('movement_rate', 0):.1f}%")
        self.metrics["data_quality"].setText(f"{stats.get('data_quality', 0):.1f}%")
        self.metrics["avg_magnitude"].setText(f"{stats.get('avg_movement_magnitude', 0):.1f} px")
        
        movement_count = stats.get('movement_count', 0)
        entropy_quality = min(100, movement_count / 2)
        self.entropy_bar.setValue(int(entropy_quality))
        self.entropy_value.setText(f"{int(entropy_quality)}%")
        
        if movement_count < 20:
            self.entropy_label.setText("⚠️ Соберите больше движений")
            self.entropy_label.setStyleSheet("font-size: 12px; color: #F44336;")
        elif movement_count < 50:
            self.entropy_label.setText("📈 Хорошо, продолжайте")
            self.entropy_label.setStyleSheet("font-size: 12px; color: #FF9800;")
        else:
            self.entropy_label.setText("✅ Отличное качество энтропии")
            self.entropy_label.setStyleSheet("font-size: 12px; color: #4CAF50;")


class AudioStatisticsWidget(QWidget):
    """
    Виджет статистики звука — отображает реальные данные
    из VoiceEntropyCollector (передаётся снаружи через set_collector).
    """

    def __init__(self):
        super().__init__()
        self._collector: Optional[VoiceEntropyCollector] = None
        self.init_ui()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._refresh_from_collector)

    def set_collector(self, collector: VoiceEntropyCollector):
        """Привязывает к реальному коллектору."""
        self._collector = collector

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(0, 0, 0, 0)

        audio_card = ModernCard(title="🎤 Статистика звука")

        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(15)

        self.metrics = {}
        metrics_data = [
            ("volume_avg",  "Громкость", "0 dB",  "#FF5722"),
            ("freq_main",   "Частота",   "0 Hz",  "#00BCD4"),
            ("snr",         "SNR",       "0 dB",  "#795548"),
        ]

        for i, (key, label, default, color) in enumerate(metrics_data):
            metric_widget = QWidget()
            metric_layout = QVBoxLayout()
            metric_layout.setSpacing(5)

            value_label = QLabel(default)
            value_label.setStyleSheet(f"""
                font-size: 22px;
                font-weight: 700;
                color: {color};
            """)
            value_label.setAlignment(Qt.AlignCenter)

            name_label = QLabel(label)
            name_label.setStyleSheet("font-size: 11px; color: #aaa;")
            name_label.setAlignment(Qt.AlignCenter)

            metric_layout.addWidget(value_label)
            metric_layout.addWidget(name_label)
            metric_widget.setLayout(metric_layout)

            row = i // 2
            col = i % 2
            metrics_grid.addWidget(metric_widget, row, col)
            self.metrics[key] = value_label

        audio_card.addWidget(metrics_grid)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #404040;")
        separator.setFixedHeight(1)
        audio_card.addWidget(separator)

        quality_widget = QWidget()
        quality_layout = QVBoxLayout()
        quality_layout.setSpacing(8)

        quality_header = QHBoxLayout()
        quality_header.addWidget(QLabel("🎵 Качество сигнала"))
        self.quality_value = QLabel("0%")
        self.quality_value.setStyleSheet("font-weight: 600; color: #aaa;")
        quality_header.addWidget(self.quality_value)
        quality_layout.addLayout(quality_header)

        self.quality_bar = QProgressBar()
        self.quality_bar.setMaximum(100)
        self.quality_bar.setTextVisible(False)
        self.quality_bar.setFixedHeight(6)
        self.quality_bar.setStyleSheet("""
            QProgressBar {
                background-color: #3a3a3a;
                border-radius: 3px;
                border: none;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF5722, stop:0.5 #FF9800, stop:1 #4CAF50);
                border-radius: 3px;
            }
        """)
        quality_layout.addWidget(self.quality_bar)
        quality_widget.setLayout(quality_layout)
        audio_card.addWidget(quality_widget)

        info_widget = QWidget()
        info_layout = QHBoxLayout()
        info_layout.setSpacing(10)

        self.active_label = QLabel("🎙️ Микрофон: Не активен")
        self.active_label.setStyleSheet("font-size: 11px; color: #888;")
        info_layout.addWidget(self.active_label)

        info_layout.addStretch()

        self.vad_label = QLabel("VAD: —")
        self.vad_label.setStyleSheet("font-size: 11px; color: #888;")
        info_layout.addWidget(self.vad_label)

        info_widget.setLayout(info_layout)
        audio_card.addWidget(info_widget)

        # Строка голосовых байт пула
        self.pool_label = QLabel("Пул: 0 байт")
        self.pool_label.setStyleSheet("font-size: 10px; color: #666;")
        audio_card.addWidget(self.pool_label)

        self.silence_label = QLabel("Тишина: —")
        self.silence_label.setStyleSheet("font-size: 10px; color: #666;")
        audio_card.addWidget(self.silence_label)

        layout.addWidget(audio_card)
        self.setLayout(layout)

    def _refresh_from_collector(self):
        """Обновляет виджет реальными данными из коллектора."""
        if self._collector is None:
            return

        stats = self._collector.get_statistics()

        avg_db   = stats.get("avg_db", -60.0)
        snr_db   = stats.get("snr_db", 0.0)
        freq     = stats.get("dominant_freq", 0.0)
        quality  = stats.get("entropy_quality", 0.0)
        is_active = stats.get("is_active", False)
        pool_bytes = stats.get("pool_bytes", 0)

        self.metrics["volume_avg"].setText(f"{avg_db:.1f} dB")
        self.metrics["freq_main"].setText(f"{freq:.0f} Hz")
        self.metrics["snr"].setText(f"{snr_db:.1f} dB")

        self.quality_bar.setValue(int(quality))
        self.quality_value.setText(f"{int(quality)}%")

        self.pool_label.setText(f"Пул: {pool_bytes} байт")

        # Процент тишины и цветовая индикация порога 40%
        if self._collector.has_enough_entropy():
            silence_pct = self._collector.silence_percent() * 100
            voice_ok    = self._collector.has_enough_voice_activity()
            color  = "#4CAF50" if voice_ok else "#FF5722"
            status = "✓ голос учитывается" if voice_ok else "✗ голос отброшен"
            self.silence_label.setText(f"Тишина: {silence_pct:.1f}%  {status}")
            self.silence_label.setStyleSheet(f"font-size: 10px; color: {color};")
        else:
            self.silence_label.setText("Тишина: накапливаются данные…")
            self.silence_label.setStyleSheet("font-size: 10px; color: #666;")

        if is_active:
            self.active_label.setText("🎙️ Микрофон: Активен")
            self.active_label.setStyleSheet("font-size: 11px; color: #4CAF50;")
            self.vad_label.setText("VAD: Голос")
            self.vad_label.setStyleSheet("font-size: 11px; color: #4CAF50;")
        else:
            self.active_label.setText("🎙️ Микрофон: Тишина")
            self.active_label.setStyleSheet("font-size: 11px; color: #888;")
            self.vad_label.setText("VAD: —")
            self.vad_label.setStyleSheet("font-size: 11px; color: #888;")

    def start_monitoring(self):
        self.update_timer.start(500)
        self.active_label.setText("🎙️ Микрофон: Активен")
        self.active_label.setStyleSheet("font-size: 11px; color: #4CAF50;")

    def stop_monitoring(self):
        self.update_timer.stop()
        self.active_label.setText("🎙️ Микрофон: Не активен")
        self.active_label.setStyleSheet("font-size: 11px; color: #888;")
        self.vad_label.setText("VAD: —")
        self.vad_label.setStyleSheet("font-size: 11px; color: #888;")

    def reset(self):
        for key, lbl in self.metrics.items():
            defaults = {"volume_avg": "0 dB", "freq_main": "0 Hz",
                        "snr": "0 dB"}
            lbl.setText(defaults.get(key, "0"))
        self.quality_bar.setValue(0)
        self.quality_value.setText("0%")
        self.pool_label.setText("Пул: 0 байт")
        self.silence_label.setText("Тишина: —")
        self.silence_label.setStyleSheet("font-size: 10px; color: #666;")


class EyeTrackerUI(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.camera_thread = CameraThread()
        self.last_generated_key = None
        self.last_raw_pool = None
        self.gaze_points = deque(maxlen=1000)
        
        self.star_field = StarField(self)
        self.star_field.star_died.connect(self.star_field.handle_star_died)
        
        self.instruction_window = None
        self.audio_visualizer = AudioVisualizerWidget()
        
        self.init_ui()
        self.connect_signals()
        # Привязываем AudioStatisticsWidget к реальному коллектору
        self.audio_stats_widget.set_collector(self.audio_visualizer.voice_collector)
        self.check_dependencies()
    
    def init_ui(self):
        self.setWindowTitle("EyeTracker")
        self.setGeometry(100, 100, 1440, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(24)
        
        # Левая панель
        left_panel = QWidget()
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Заголовок
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("👁 EyeTracker")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #e0e0e0;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        instruction_btn = IconButton("📚")
        instruction_btn.clicked.connect(self.show_instruction)
        header_layout.addWidget(instruction_btn)
        
        left_layout.addWidget(header)
        
        # Карточка выбора источников энтропии
        source_card = ModernCard(title="📡 Источники энтропии")

        self.source_toggle = TripleToggle()
        self._entropy_source = "both"   # "eye" | "voice" | "both"
        source_card.addWidget(self.source_toggle)

        self.source_info_label = QLabel("Режим: взгляд + голос")
        self.source_info_label.setStyleSheet("font-size: 11px; color: #4CAF50; font-weight: 600;")
        self.source_info_label.setAlignment(Qt.AlignCenter)
        source_card.addWidget(self.source_info_label)

        left_layout.addWidget(source_card)

        # Карточка управления
        control_card = ModernCard(title="🎮 Управление")
        
        self.status_label = QLabel("⏸ Готов к работе")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 500;
            color: #c0c0c0;
            padding: 12px;
            background-color: #3a3a3a;
            border-radius: 8px;
        """)
        control_card.addWidget(self.status_label)
        
        self.start_btn = ModernButton("▶ Начать трекинг", primary=True)
        control_card.addWidget(self.start_btn)
        
        self.stop_btn = ModernButton("⏹ Остановить")
        self.stop_btn.setEnabled(False)
        control_card.addWidget(self.stop_btn)
        
        self.reset_btn = ModernButton("🔄 Сбросить данные")
        control_card.addWidget(self.reset_btn)
        
        left_layout.addWidget(control_card)
        
        # Карточка триггеров
        triggers_card = ModernCard(title="🎲 Стимулы")
        
        trigger_buttons = QHBoxLayout()
        trigger_buttons.setSpacing(10)
        
        self.youtube_btn = ModernButton("▶ YouTube", accent=True)
        trigger_buttons.addWidget(self.youtube_btn)
        
        self.browser_btn = ModernButton("🌐 Браузер")
        trigger_buttons.addWidget(self.browser_btn)
        
        triggers_card.addWidget(trigger_buttons)
        
        self.triggers_toggle = ModernToggle(text="Случайные триггеры")
        triggers_card.addWidget(self.triggers_toggle)
        
        left_layout.addWidget(triggers_card)
        
        # Карточка ключа
        key_card = ModernCard(title="🔐 Генерация ключа")
        
        key_buttons = QHBoxLayout()
        key_buttons.setSpacing(10)
        
        self.key_128_btn = ModernButton("128 бит")
        key_buttons.addWidget(self.key_128_btn)
        
        self.key_256_btn = ModernButton("256 бит", accent=True)
        key_buttons.addWidget(self.key_256_btn)
        
        key_card.addWidget(key_buttons)
        
        self.save_btn = ModernButton("💾 Сохранить ключ")
        self.save_btn.setEnabled(False)
        key_card.addWidget(self.save_btn)
        
        left_layout.addWidget(key_card)
        left_layout.addStretch()
        
        # Центральная панель
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setSpacing(20)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        # Карточка с видео и аудио
        media_card = ModernCard(title="📹 Камера и аудио")
        
        media_split_layout = QHBoxLayout()
        media_split_layout.setSpacing(15)
        
        # Видео виджет
        self.video_widget = VideoWidget()
        media_split_layout.addWidget(self.video_widget)
        
        # Аудио виджет
        audio_widget = QWidget()
        audio_widget.setMinimumSize(400, 300)
        audio_layout = QVBoxLayout(audio_widget)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        audio_layout.setSpacing(5)
        
        audio_label = QLabel("🎤 Аудио визуализация")
        audio_label.setStyleSheet("font-size: 12px; color: #aaa; padding: 5px;")
        audio_layout.addWidget(audio_label)
        
        audio_layout.addWidget(self.audio_visualizer)
        
        media_split_layout.addWidget(audio_widget)
        
        media_card.addWidget(media_split_layout)
        center_layout.addWidget(media_card)
        
        # Отображение ключа
        key_display_card = ModernCard(title="🔑 Сгенерированный ключ")
        
        self.key_display = QTextEdit()
        self.key_display.setReadOnly(True)
        self.key_display.setMaximumHeight(80)
        self.key_display.setPlaceholderText("Здесь появится сгенерированный ключ...")
        self.key_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #4CAF50;
                border: 1px solid #404040;
                border-radius: 8px;
                font-family: 'SF Mono', 'Monaco', monospace;
                font-size: 13px;
                padding: 12px;
            }
        """)
        key_display_card.addWidget(self.key_display)
        
        key_info_layout = QHBoxLayout()
        self.key_size_label = QLabel("Размер: —")
        self.key_size_label.setStyleSheet("font-size: 12px; color: #aaa;")
        key_info_layout.addWidget(self.key_size_label)
        key_info_layout.addStretch()
        
        self.copy_key_btn = IconButton("📋", "Копировать")
        self.copy_key_btn.setEnabled(False)
        self.copy_key_btn.setFixedHeight(36)
        key_info_layout.addWidget(self.copy_key_btn)

        self.save_pool_btn = IconButton("⬇", "Скачать пул энтропии (.bin)")
        self.save_pool_btn.setEnabled(False)
        self.save_pool_btn.setFixedHeight(36)
        key_info_layout.addWidget(self.save_pool_btn)
        
        key_display_card.addWidget(key_info_layout)
        center_layout.addWidget(key_display_card)
        
        # Правая панель
        right_panel = QWidget()
        right_panel.setFixedWidth(340)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.stats_widget = StatisticsWidget()
        right_layout.addWidget(self.stats_widget)
        
        self.audio_stats_widget = AudioStatisticsWidget()
        right_layout.addWidget(self.audio_stats_widget)
        
        right_layout.addStretch()
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel, 1)
        main_layout.addWidget(right_panel)
        
        # Статус-бар
        self.key_info_label = QLabel("Готов к работе")
        self.key_info_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self.statusBar().addWidget(self.key_info_label)
        
        # Показываем заглушку при запуске
        self.video_widget.show_placeholder()
    
    def connect_signals(self):
        self.source_toggle.changed.connect(self._on_source_toggle)
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.reset_btn.clicked.connect(self.reset_data)
        self.key_128_btn.clicked.connect(lambda: self.generate_key(128))
        self.key_256_btn.clicked.connect(lambda: self.generate_key(256))
        self.save_btn.clicked.connect(self.save_key)
        self.copy_key_btn.clicked.connect(self.copy_key_to_clipboard)
        self.save_pool_btn.clicked.connect(self.save_raw_pool)
        
        self.youtube_btn.clicked.connect(self.open_youtube)
        self.browser_btn.clicked.connect(self.open_chrome)
        self.triggers_toggle.toggled.connect(self.toggle_triggers)
        
        self.camera_thread.frame_ready.connect(self.update_video)
        self.camera_thread.stats_updated.connect(self.stats_widget.update_stats)
        self.camera_thread.error_occurred.connect(self.handle_error)
        self.camera_thread.camera_started.connect(self.on_camera_started)
    
    def check_dependencies(self):
        if not MEDIAPIPE_AVAILABLE:
            QMessageBox.critical(self, "Ошибка", "MediaPipe не установлен!")
    
    def show_instruction(self):
        if not self.instruction_window:
            self.instruction_window = InstructionWidget()
        self.instruction_window.show()
    
    def open_youtube(self):
        webbrowser.open("https://www.youtube.com")
    
    def open_chrome(self):
        webbrowser.open("https://www.google.com")
    
    def toggle_triggers(self, state):
        if state:
            self.star_field.start_field()
        else:
            self.star_field.stop_field()
    
    def on_camera_started(self):
        pass
    
    @pyqtSlot(object, object)
    def update_video(self, frame, gaze_data):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.video_widget.setPixmap(pixmap)
        
        if gaze_data:
            self.gaze_points.append((gaze_data['x'], gaze_data['y']))
            
            if gaze_data['movement_detected']:
                self.status_label.setText(f"👁 Движение: {gaze_data['movement_magnitude']:.1f}px")
                self.status_label.setStyleSheet("""
                    font-size: 14px; font-weight: 500; color: #81C784;
                    padding: 12px; background-color: #2E3B2E; border-radius: 8px;
                """)
            else:
                self.status_label.setText("⏸ Покой")
                self.status_label.setStyleSheet("""
                    font-size: 14px; font-weight: 500; color: #FFB74D;
                    padding: 12px; background-color: #3E3524; border-radius: 8px;
                """)
    
    def _on_source_toggle(self, pos: int):
        """Вызывается при переключении тумблера источников."""
        self._entropy_source = self.source_toggle.source()
        labels = {
            "eye":   ("Режим: только взгляд",  "#64B5F6"),
            "voice": ("Режим: только голос",   "#FF9800"),
            "both":  ("Режим: взгляд + голос", "#4CAF50"),
        }
        text, color = labels[self._entropy_source]
        self.source_info_label.setText(text)
        self.source_info_label.setStyleSheet(
            f"font-size: 11px; color: {color}; font-weight: 600;"
        )

    def start_tracking(self):
        self.video_widget.hide_placeholder()
        self.video_widget.show_loading()
        
        self.camera_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("🎯 Запуск камеры...")
        self.status_label.setStyleSheet("""
            font-size: 14px; font-weight: 500; color: #64B5F6;
            padding: 12px; background-color: #1A2A3A; border-radius: 8px;
        """)
        
        # Блокируем тумблер во время трекинга
        self.source_toggle.setEnabled(False)

        if self._entropy_source in ("voice", "both"):
            self.audio_visualizer.start_recording()
            self.audio_stats_widget.start_monitoring()
    
    def stop_tracking(self):
        self.camera_thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("⏸ Трекинг остановлен")
        self.status_label.setStyleSheet("""
            font-size: 14px; font-weight: 500; color: #c0c0c0;
            padding: 12px; background-color: #3a3a3a; border-radius: 8px;
        """)
        
        self.video_widget.show_placeholder()
        self.video_widget.first_frame_received = False
        self.video_widget.is_camera_active = False
        
        if self._entropy_source in ("voice", "both"):
            self.audio_visualizer.stop_recording()
            self.audio_stats_widget.stop_monitoring()

        # Разблокируем тумблер
        self.source_toggle.setEnabled(True)
        
        if self.star_field.is_active:
            self.triggers_toggle.state = False
            self.triggers_toggle.update()
            self.star_field.stop_field()
    
    def reset_data(self):
        self.camera_thread.reset_data()
        self.gaze_points.clear()
        self.audio_visualizer.reset()
        self.audio_stats_widget.reset()
        self.status_label.setText("🔄 Данные сброшены")
    
    def generate_key(self, bits: int):
        if not self.camera_thread.isRunning():
            QMessageBox.warning(self, "Предупреждение", "Сначала запустите трекинг!")
            return
        
        stats = self.camera_thread.tracker.get_movement_statistics()
        if stats['movement_count'] < 20:
            QMessageBox.warning(self, "Недостаточно данных", 
                f"Необходимо минимум 20 движений, сейчас: {stats['movement_count']}")
            return

        from VoiceEntropy import derive_key, combine_entropy

        # ── Собираем данные согласно выбранному режиму источников ─────────
        voice_collector = self.audio_visualizer.voice_collector
        use_eye   = self._entropy_source in ("eye",   "both")
        use_voice = self._entropy_source in ("voice", "both")

        # Глазные байты
        if use_eye:
            eye_key_bytes = self.camera_thread.generate_key(bits)
            eye_bytes = eye_key_bytes if eye_key_bytes else b""
        else:
            eye_bytes = b""

        # Голосовые байты — только если режим включает голос
        # и микрофон реально работал с достаточной активностью
        if use_voice:
            mic_ok      = voice_collector.microphone_active()
            voice_ok    = mic_ok and voice_collector.has_enough_voice_activity()
            voice_bytes = voice_collector.get_entropy_bytes() if voice_ok else b""

            if not mic_ok:
                source_info = " [микрофон недоступен]"
                use_voice = False
            elif not voice_ok:
                silence_pct = int(voice_collector.silence_percent() * 100)
                source_info = f" [голос отброшен — тишина {silence_pct}% > 40%]"
                use_voice = False
        else:
            voice_bytes = b""

        # ── Строим raw_pool из доступных источников ───────────────────────
        if use_eye and use_voice and eye_bytes and voice_bytes:
            raw_pool    = combine_entropy(eye_bytes, voice_bytes)
            source_info = " [👁+🎤]"
        elif use_eye and eye_bytes:
            raw_pool    = eye_bytes
            source_info = " [👁]" if self._entropy_source == "eye" else " [👁 — голос недоступен]"
        elif use_voice and voice_bytes:
            raw_pool    = voice_bytes
            source_info = " [🎤]"
        else:
            QMessageBox.critical(self, "Ошибка", "Нет данных от выбранных источников")
            return

        # ── Проверка минимального размера пула (SP800-90B калибровка) ────────
        min_bytes = MIN_POOL_BYTES_128 if bits == 128 else MIN_POOL_BYTES_256
        if len(raw_pool) < min_bytes:
            needed   = min_bytes - len(raw_pool)
            QMessageBox.warning(
                self, "Недостаточно энтропии",
                f"Пул слишком мал для надёжной генерации {bits}-бит ключа.\n\n"
                f"Накоплено:    {len(raw_pool)} байт\n"
                f"Необходимо:  {min_bytes} байт "
                f"(на основе SP800-90B, H_min = {H_MIN_BITS_PER_BYTE} бит/байт)\n"
                f"Не хватает:  {needed} байт\n\n"
                f"Продолжайте двигать глазами"
                + (" и говорить" if self._entropy_source in ("voice", "both") else "")
                + "."
            )
            return

        key = derive_key(raw_pool, bits) if raw_pool else b""

        if key and len(key) > 0:
            self.last_generated_key = key
            self.last_raw_pool = raw_pool
            hex_key = key.hex()

            self.key_display.setText(hex_key)
            self.key_size_label.setText(f"Размер: {len(key)} байт ({bits} бит){source_info}")
            self.save_btn.setEnabled(True)
            self.copy_key_btn.setEnabled(True)
            self.save_pool_btn.setEnabled(True)
            
            filename = f"eye_voice_key_{bits}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
            with open(filename, "wb") as f:
                f.write(key)
            
            QMessageBox.information(self, "Успех",
                f"Ключ {bits} бит сгенерирован{source_info}\nСохранён: {filename}")
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось сгенерировать ключ")
    
    def copy_key_to_clipboard(self):
        if self.last_generated_key:
            QApplication.clipboard().setText(self.last_generated_key.hex())
            QMessageBox.information(self, "Скопировано", "Ключ скопирован в буфер обмена")
    
    def save_key(self):
        if not self.last_generated_key:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить ключ",
            f"eye_key_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin",
            "Binary Files (*.bin)"
        )
        
        if filename:
            with open(filename, "wb") as f:
                f.write(self.last_generated_key)
            QMessageBox.information(self, "Сохранено", f"Ключ сохранен в:\n{filename}")
    
    def save_raw_pool(self):
        if not self.last_raw_pool:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить сырой пул энтропии",
            f"entropy_pool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin",
            "Binary Files (*.bin)"
        )

        if filename:
            with open(filename, "wb") as f:
                f.write(self.last_raw_pool)
            QMessageBox.information(
                self, "Сохранено",
                f"Пул энтропии сохранён:\n{filename}\nРазмер: {len(self.last_raw_pool)} байт"
            )

    def handle_error(self, error_msg: str):
        QMessageBox.critical(self, "Ошибка", error_msg)
        self.video_widget.hide_loading()
        self.video_widget.show_placeholder()
    
    def closeEvent(self, event):
        if self.camera_thread.isRunning():
            reply = QMessageBox.question(self, 'Подтверждение',
                'Трекинг активен. Остановить и выйти?',
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.star_field.stop_field()
                self.audio_visualizer.stop_recording()
                self.camera_thread.stop()
                event.accept()
            else:
                event.ignore()
        else:
            self.star_field.stop_field()
            self.audio_visualizer.stop_recording()
            event.accept()


def main():
    app = QApplication(sys.argv)
    DarkTheme.apply(app)
    
    window = EyeTrackerUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()