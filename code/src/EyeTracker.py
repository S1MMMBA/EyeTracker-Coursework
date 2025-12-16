import cv2
import numpy as np
import math
import time
import json
from collections import deque, Counter
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import mediapipe as mp
    from mediapipe.python.solutions.face_mesh import FaceMesh
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe не установлен. Установите: pip install mediapipe")

class AdvancedGazeTracker:
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Инициализация MediaPipe с улучшенными параметрами
        self._initialize_mediapipe()
        
        # Параметры трекинга
        self.movement_threshold = 15  # пикселей для детекции движения 15
        self.min_movement_interval = 0.05  # минимальный интервал между движениями 0.15
        self.eye_openness_threshold = 0.4  # порог открытости глаз 0.4
        
        # Хранилище данных (ТОЛЬКО движения)
        # self.movement_events = deque(maxlen=200)  # ограниченная история движений
        # self.gaze_history = deque(maxlen=30)  # краткая история для вычисления движения
        # self.eye_state_history = deque(maxlen=50)

        self.movement_events = deque()  # ограниченная история движений
        self.gaze_history = deque()  # краткая история для вычисления движения
        self.eye_state_history = deque()
        
        # Статистика
        self.frame_count = 0
        self.last_movement_time = 0
        self.consecutive_still_frames = 0
        self.last_valid_gaze = None
        
        # Калибровка
        self.calibration_data = {}
        self.is_calibrated = False
        
        print(" Продвинутый трекер взгляда инициализирован (только MediaPipe)")
        print(f" Разрешение экрана: {screen_width}x{screen_height}")
    
    def _initialize_mediapipe(self):
        """Инициализация MediaPipe с оптимизированными параметрами"""
        if not MEDIAPIPE_AVAILABLE:
            return
            
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            static_image_mode=False  # Оптимизация для видео
        )
        
        # Определение ключевых точек глаз
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Точки для определения открытости глаз
        self.LEFT_EYE_VERTICAL = [159, 145]  # верх и низ левого глаза
        self.RIGHT_EYE_VERTICAL = [386, 374]  # верх и низ правого глаза
        self.LEFT_EYE_HORIZONTAL = [33, 133]  # левый и правый угол левого глаза
        self.RIGHT_EYE_HORIZONTAL = [362, 263] # левый и правый угол правого глаза
    
    def process_frame(self, frame):
        """
        Обработка кадра - возвращает данные ТОЛЬКО при обнаружении движения
        """
        if not MEDIAPIPE_AVAILABLE:
            return frame, None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        movement_detected = False
        gaze_data = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. Проверка состояния глаз
                eye_state = self._analyze_eye_state(face_landmarks)
                self.eye_state_history.append(eye_state)
                
                # Пропускаем кадр если глаза закрыты
                if not eye_state['eyes_open']:
                    self._draw_status(frame, "EYES CLOSED", (0, 0, 255))
                    return frame, None
                
                # 2. Получение текущей позиции взгляда
                current_gaze = self._calculate_advanced_gaze(face_landmarks, frame.shape)
                
                # 3. Проверка движения
                movement_detected, movement_info = self._detect_significant_movement(current_gaze)
                
                if movement_detected:
                    # СОХРАНЯЕМ ДАННЫЕ ТОЛЬКО ПРИ ДВИЖЕНИИ
                    gaze_data = {
                        'x': float(current_gaze[0]),
                        'y': float(current_gaze[1]),
                        'timestamp': time.time(),
                        'valid': True,
                        'movement_detected': True,
                        'movement_magnitude': movement_info['magnitude'],
                        'movement_direction': movement_info['direction'],
                        'eye_openness': eye_state['avg_openness'],
                        'attention_score': self._calculate_attention_score(eye_state)
                    }
                    
                    self.movement_events.append(gaze_data)
                    self.last_movement_time = time.time()
                    self.consecutive_still_frames = 0
                    self.last_valid_gaze = current_gaze
                    
                else:
                    # Нет движения - не сохраняем данные
                    self.consecutive_still_frames += 1
                    gaze_data = None
                
                # Визуализация
                self._draw_advanced_visualization(frame, face_landmarks, movement_detected, current_gaze)
                status_text = "MOVEMENT" if movement_detected else f"STILL ({self.consecutive_still_frames})"
                status_color = (0, 255, 0) if movement_detected else (0, 165, 255)
                self._draw_status(frame, status_text, status_color)
        
        self.frame_count += 1
        return frame, gaze_data
    
    def _analyze_eye_state(self, landmarks):
        """Точный анализ состояния глаз"""
        left_openness = self._calculate_eye_openness(landmarks, 'left')
        right_openness = self._calculate_eye_openness(landmarks, 'right')
        
        # Строгая проверка открытости глаз
        eyes_open = (left_openness > self.eye_openness_threshold and 
                    right_openness > self.eye_openness_threshold and
                    abs(left_openness - right_openness) < 0.15)  # проверка симметрии
        
        return {
            'eyes_open': eyes_open,
            'left_openness': left_openness,
            'right_openness': right_openness,
            'avg_openness': (left_openness + right_openness) / 2,
            'symmetry': 1.0 - abs(left_openness - right_openness),
            'timestamp': time.time()
        }
    
    def _calculate_eye_openness(self, landmarks, eye):
        """Расчет открытости глаза на основе ключевых точек"""
        if eye == 'left':
            vertical_indices = self.LEFT_EYE_VERTICAL
            horizontal_indices = self.LEFT_EYE_HORIZONTAL
        else:
            vertical_indices = self.RIGHT_EYE_VERTICAL
            horizontal_indices = self.RIGHT_EYE_HORIZONTAL
        
        try:
            # Вертикальное расстояние (открытость)
            top_point = landmarks.landmark[vertical_indices[0]]
            bottom_point = landmarks.landmark[vertical_indices[1]]
            vertical_dist = abs(top_point.y - bottom_point.y)
            
            # Горизонтальное расстояние (ширина глаза)
            left_point = landmarks.landmark[horizontal_indices[0]]
            right_point = landmarks.landmark[horizontal_indices[1]]
            horizontal_dist = abs(left_point.x - right_point.x)
            
            if horizontal_dist == 0:
                return 0.0
            
            # Отношение высоты к ширины
            openness_ratio = vertical_dist / horizontal_dist
            return min(1.0, openness_ratio * 2.0)  # Нормализация У человека ширина глаза обычно больше высоты. Отношение высоты к ширине у открытого глаза составляет примерно 0.3-0.5
            # Открытый глаз: 0.5 × 2.0 = 1.0; Прищуренный: 0.25 × 2.0 = 0.5
        except Exception as e:
            return 0.0
    
    def _calculate_advanced_gaze(self, landmarks, frame_shape):
        """Улучшенное вычисление направления взгляда"""
        h, w = frame_shape[:2]
        
        # Метод 1: Центры глаз
        left_center = self._calculate_eye_center(landmarks, self.LEFT_EYE_INDICES)
        right_center = self._calculate_eye_center(landmarks, self.RIGHT_EYE_INDICES)
        
        # Метод 2: Относительное положение зрачков
        pupil_offset = self._estimate_pupil_offset(landmarks)
        
        # Комбинируем оба метода
        gaze_x = (left_center[0] + right_center[0]) / 2 + pupil_offset[0] * 0.3
        gaze_y = (left_center[1] + right_center[1]) / 2 + pupil_offset[1] * 0.3
        
        # Преобразование в координаты экрана
        screen_x = gaze_x * self.screen_width
        screen_y = gaze_y * self.screen_height
        
        return (screen_x, screen_y)
    
    def _calculate_eye_center(self, landmarks, eye_indices):
        """Вычисление центра глаза"""
        points = []
        for idx in eye_indices:
            point = landmarks.landmark[idx]
            points.append((point.x, point.y))
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return center_x, center_y
    
    def _estimate_pupil_offset(self, landmarks):
        """Оценка смещения зрачков для определения направления взгляда"""
        try:
            # Для левого глаза
            left_center = self._calculate_eye_center(landmarks, self.LEFT_EYE_INDICES)
            left_iris = landmarks.landmark[468]  # Примерная точка радужки
            
            # Для правого глаза  
            right_center = self._calculate_eye_center(landmarks, self.RIGHT_EYE_INDICES)
            right_iris = landmarks.landmark[473]  # Примерная точка радужки
            
            # Среднее смещение
            offset_x = ((left_iris.x - left_center[0]) + (right_iris.x - right_center[0])) / 2
            offset_y = ((left_iris.y - left_center[1]) + (right_iris.y - right_center[1])) / 2
            
            return (offset_x, offset_y)
        except:
            return (0, 0)
    
    def _detect_significant_movement(self, current_gaze):
        """Детекция значительного движения глаз"""
        current_time = time.time()
        
        # Проверяем временной интервал
        time_since_last_movement = current_time - self.last_movement_time
        if time_since_last_movement < self.min_movement_interval:
            return False, {}
        
        # Если это первое движение
        if self.last_valid_gaze is None:
            self.last_valid_gaze = current_gaze
            self.gaze_history.append(current_gaze)
            return True, {'magnitude': 0, 'direction': 0}
        
        # Вычисляем величину движения
        movement_magnitude = math.sqrt(
            (current_gaze[0] - self.last_valid_gaze[0])**2 + 
            (current_gaze[1] - self.last_valid_gaze[1])**2
        )
        
        # Вычисляем направление движения
        dx = current_gaze[0] - self.last_valid_gaze[0]
        dy = current_gaze[1] - self.last_valid_gaze[1]
        movement_direction = math.atan2(dy, dx)
        
        # Проверяем порог движения
        if movement_magnitude > self.movement_threshold:
            self.gaze_history.append(current_gaze)
            return True, {
                'magnitude': movement_magnitude,
                'direction': movement_direction
            }
        
        return False, {}
    
    def _calculate_attention_score(self, eye_state):
        """Оценка уровня внимания на основе состояния глаз"""
        # Комбинируем несколько факторов
        openness_score = eye_state['avg_openness']
        symmetry_score = eye_state['symmetry']
        stability_score = 1.0 - min(1.0, self.consecutive_still_frames / 50.0)
        
        # Взвешенная сумма
        attention_score = (
            openness_score * 0.4 +
            symmetry_score * 0.3 + 
            stability_score * 0.3
        )
        
        return min(1.0, max(0.0, attention_score))
    
    def _draw_advanced_visualization(self, frame, landmarks, movement_detected, current_gaze):
        """Улучшенная визуализация"""
        h, w = frame.shape[:2]
        
        # Цвет в зависимости от движения
        color = (0, 255, 0) if movement_detected else (0, 165, 255)  # Зеленый или оранжевый
        
        # Рисуем ключевые точки глаз
        for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
            point = landmarks.landmark[idx]
            x = int(point.x * w)
            y = int(point.y * h)
            cv2.circle(frame, (x, y), 2, color, -1)
        
        # Рисуем точку взгляда
        gaze_x = int(current_gaze[0] * w / self.screen_width)
        gaze_y = int(current_gaze[1] * h / self.screen_height)
        
        if movement_detected:
            cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (gaze_x, gaze_y), 15, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (gaze_x, gaze_y), 6, (0, 165, 255), -1)
        
        # Линия от центра к точке взгляда
        center_x, center_y = w // 2, h // 2
        cv2.line(frame, (center_x, center_y), (gaze_x, gaze_y), color, 1)
    
    def _draw_status(self, frame, status, color):
        """Отображение статуса"""
        stats = self.get_movement_statistics()
        
        # Исправляем условное выражение
        if self.eye_state_history:
            eye_status = 'OPEN' if self.eye_state_history[-1]['eyes_open'] else 'CLOSED'
        else:
            eye_status = 'UNKNOWN'
        
        info_text = [
            f"Status: {status}",
            f"Movements: {stats['movement_count']}",
            f"Still Frames: {self.consecutive_still_frames}",
            f"Eye State: {eye_status}",
            f"Data Quality: {stats['data_quality']:.1f}%"
        ]
        
        for i, text in enumerate(info_text):
            y_position = 30 + i * 25
            cv2.putText(frame, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    def get_movement_statistics(self):
        """Получение статистики движений"""
        movement_count = len(self.movement_events)
        total_frames = self.frame_count
        
        if total_frames == 0:
            return {
                'movement_count': 0,
                'movement_rate': 0.0,
                'data_quality': 0.0,
                'avg_movement_magnitude': 0.0
            }
        
        # Вычисляем среднюю величину движения
        magnitudes = [event['movement_magnitude'] for event in self.movement_events]
        avg_magnitude = np.mean(magnitudes) if magnitudes else 0.0
        
        # Качество данных (процент кадров с открытыми глазами)
        valid_eye_states = [state for state in self.eye_state_history if state['eyes_open']]
        data_quality = (len(valid_eye_states) / len(self.eye_state_history)) * 100 if self.eye_state_history else 0.0
        
        return {
            'movement_count': movement_count,
            'movement_rate': (movement_count / total_frames) * 100,
            'data_quality': data_quality,
            'avg_movement_magnitude': avg_magnitude,
            'total_frames': total_frames
        }


class MovementBasedCryptoGenerator:
    """
    КРИПТОГЕНЕРАТОР на основе движений глаз
    Генерирует энтропию ТОЛЬКО из событий движения
    """
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.entropy_pool = deque(maxlen=1000)
        
    def extract_movement_entropy(self):
        """Извлечение энтропии из движений глаз"""
        if len(self.tracker.movement_events) < 5:
            return []
        
        # Берем последние движения
        recent_movements = list(self.tracker.movement_events)[-10:]
        bits = []
        
        for movement in self.tracker.movement_events:
            # Извлекаем энтропию из различных параметров движения
            movement_bits = self._extract_movement_bits(movement)
            timing_bits = self._extract_timing_bits(movement)
            physiological_bits = self._extract_physiological_bits(movement)
            
            bits.extend(movement_bits + timing_bits + physiological_bits)
        
        return bits
    
    def _extract_movement_bits(self, movement):
        """Извлечение битов из параметров движения"""
        bits = []
        
        # Координаты (младшие биты)
        x_int = int(movement['x'] * 1000) % 256
        y_int = int(movement['y'] * 1000) % 256
        
        x_bits = [int(bit) for bit in format(x_int, '08b')[-4:]]  # 4 младших бита
        y_bits = [int(bit) for bit in format(y_int, '08b')[-4:]]  # 4 младших бита
        
        # Величина движения
        magnitude_int = int(movement['movement_magnitude'] * 100) % 16
        magnitude_bits = [int(bit) for bit in format(magnitude_int, '04b')]
        
        # Направление (квантованное)
        direction = movement['movement_direction']
        direction_quantized = int((direction + math.pi) / (2 * math.pi) * 8) % 8
        direction_bits = [int(bit) for bit in format(direction_quantized, '03b')]
        
        bits.extend(x_bits + y_bits + magnitude_bits + direction_bits)
        return bits
    
    def _extract_timing_bits(self, movement):
        """Извлечение битов из временных параметров"""
        bits = []
        
        # Микросекунды времени
        timestamp = movement['timestamp']
        microseconds = int((timestamp - int(timestamp)) * 1e6)
        
        # 4 младших бита из микросекунд
        time_bits = [int(bit) for bit in format(microseconds % 16, '04b')]
        bits.extend(time_bits)
        
        return bits
    
    def _extract_physiological_bits(self, movement):
        """Извлечение битов из физиологических параметров"""
        bits = []
        
        # Открытость глаз
        openness_int = int(movement['eye_openness'] * 100) % 8
        openness_bits = [int(bit) for bit in format(openness_int, '03b')]
        
        # Уровень внимания
        attention_int = int(movement['attention_score'] * 100) % 8
        attention_bits = [int(bit) for bit in format(attention_int, '03b')]
        
        bits.extend(openness_bits + attention_bits)
        return bits
    
    def generate_secure_bits(self, num_bits=256):
        """Генерация безопасных битов"""
        movement_count = len(self.tracker.movement_events)
        
        if movement_count < 20:
            print(f" Недостаточно движений: {movement_count}/20")
            return []
        
        print(f" Генерация {num_bits} битов из {movement_count} движений...")
        
        bits = []
        attempts = 0
        max_attempts = 50
        with open("newbits_data.bin", "wb") as f:
            while len(bits) < num_bits and attempts < max_attempts:
                new_bits = self.extract_movement_entropy()
                if new_bits:
                    bits.extend(new_bits)

                for i in range(0, len(new_bits), 8):
                    chunk = new_bits[i:i+8]
                    if len(chunk) == 8:
                        byte_value = int(''.join(map(str, chunk)), 2)
                        f.write(bytes([byte_value]))

                attempts += 1
                
                # Короткая пауза для накопления новых движений
                if len(bits) < num_bits:
                    time.sleep(0.1)


        final_bits = bits[:num_bits]
        
        # Проверка качества
        if final_bits:
            ones = sum(final_bits)
            proportion = ones / len(final_bits)
            print(f" Сгенерировано {len(final_bits)} битов (баланс: {proportion:.3f})")
            print(f" Массивы new_bits сохранены в newbits_data.bin")
        
        return final_bits
    
    def bits_to_bytes(self, bits):
        """Преобразование битов в байты"""
        if len(bits) % 8 != 0:
            bits = bits[:-(len(bits) % 8)]  # Обрезаем до кратного 8
        
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            byte_value = int(''.join(map(str, byte_bits)), 2)
            byte_array.append(byte_value)
        
        return bytes(byte_array)


def main():
    """Основная функция улучшенного трекера"""
    if not MEDIAPIPE_AVAILABLE:
        print(" MediaPipe не установлен. Установите: pip install mediapipe")
        return
    
    # Инициализация
    tracker = AdvancedGazeTracker()
    crypto_generator = MovementBasedCryptoGenerator(tracker)
    
    # Настройка камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Не удалось открыть камеру")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(" УПРАВЛЕНИЕ:")
    print("   Q или ESC - Выход")
    print("   S - Статистика")
    print("   B - Генерация битов (128 бит)")
    print("   K - Генерация ключа (256 бит)")
    print("   R - Сброс данных")
    print("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Обработка кадра
            processed_frame, gaze_data = tracker.process_frame(frame)
            
            # Отображение
            cv2.imshow('Advanced Eye Tracker - Q to quit', processed_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                stats = tracker.get_movement_statistics()
                print("\n СТАТИСТИКА:")
                print(f"   Всего кадров: {stats['total_frames']}")
                print(f"   Движений: {stats['movement_count']}")
                print(f"   Процент движения: {stats['movement_rate']:.1f}%")
                print(f"   Качество данных: {stats['data_quality']:.1f}%")
                print(f"   Средняя величина движения: {stats['avg_movement_magnitude']:.1f}px")
            elif key == ord('b'):
                print("\n Генерация 128 случайных битов...")
                bits = crypto_generator.generate_secure_bits(128)
                if bits:
                    # Простая проверка случайности
                    ones = sum(bits)
                    proportion = ones / len(bits)
                    print(f" Сгенерировано {len(bits)} битов")
                    print(f"   Баланс: {ones}/1, {len(bits)-ones}/0 ({proportion:.3f})")
                    
                    # Сохранение битов
                    with open("random_bits.txt", "w") as f:
                        f.write(''.join(map(str, bits)))
                    print("   Биты сохранены в random_bits.txt")
                else:
                    print(" Не удалось сгенерировать биты")
            elif key == ord('k'):
                print("\n Генерация криптографического ключа 256 бит...")
                bits = crypto_generator.generate_secure_bits(256)
                if bits and len(bits) == 256:
                    key_bytes = crypto_generator.bits_to_bytes(bits)
                    
                    with open("eye_crypto_key.bin", "wb") as f:
                        f.write(key_bytes)
                    
                    print(f" Ключ сохранен: {len(key_bytes)} байт")
                    print(f" HEX: {key_bytes.hex()[:32]}...")
                    
                    # Проверка случайности
                    ones = sum(bits)
                    proportion = ones / len(bits)
                    print(f"   Баланс битов: {proportion:.3f} (идеально 0.5)")
                else:
                    print(f" Не удалось сгенерировать ключ ({len(bits) if bits else 0}/256 бит)")
            elif key == ord('r'):
                tracker.movement_events.clear()
                tracker.gaze_history.clear()
                tracker.eye_state_history.clear()
                tracker.consecutive_still_frames = 0
                tracker.last_movement_time = 0
                print(" Данные сброшены")
    
    except KeyboardInterrupt:
        print("\n Прервано пользователем")
    except Exception as e:
        print(f"\n Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Финальный отчет
        stats = tracker.get_movement_statistics()
        print(f"\n ФИНАЛЬНАЯ СТАТИСТИКА:")
        print(f"   Обработано кадров: {stats['total_frames']}")
        print(f"   Зарегистрировано движений: {stats['movement_count']}")
        print(f"   Эффективность сбора: {stats['movement_rate']:.1f}%")
        print(f"   Качество данных: {stats['data_quality']:.1f}%")
        
        if stats['movement_count'] >= 50:
            print("\n Автоматическая генерация финального ключа...")
            bits = crypto_generator.generate_secure_bits(256)
            if bits and len(bits) == 256:
                key_bytes = crypto_generator.bits_to_bytes(bits)
                with open("final_eye_key.bin", "wb") as f:
                    f.write(key_bytes)
                print(f" Финальный ключ сохранен: {len(key_bytes)} байт")


if __name__ == "__main__":
    main()