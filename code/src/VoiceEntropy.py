"""
VoiceEntropy.py — Сбор высокоэнтропийных данных из голоса/звука микрофона.

ОБОСНОВАНИЕ ВЫБОРА ПАРАМЕТРОВ
==============================

1. МГНОВЕННАЯ АМПЛИТУДА (временная область)
   - Образец сигнала PCM захватывается с частотой 44100 Гц
   - Младшие биты амплитуды (~2–3 бита) определяются тепловым шумом АЦП
     и акустическими флуктуациями — непредсказуемы
   - Простой, быстрый источник энтропии

2. СПЕКТРАЛЬНЫЕ КОЭФФИЦИЕНТЫ (частотная область, FFT)
   - Голос человека богат обертонами; небольшие вариации формантных частот
     уникальны для каждого произнесения
   - Фазы FFT-бинов крайне чувствительны к миллисекундным сдвигам начала
     захвата — надёжный источник
   - Берём ~16 ненулевых бинов в голосовом диапазоне 80–3400 Гц

3. МФКЦ (Mel-Frequency Cepstral Coefficients)
   - 13 коэффициентов характеризуют форму голосового тракта
   - Дробные части МФКЦ при разных произнесениях одного слова уникальны
   - Хорошо изученный параметр в биометрии голоса → высокая энтропия

4. НУЛЕВЫЕ ПЕРЕСЕЧЕНИЯ (Zero-Crossing Rate)
   - Число раз, когда сигнал меняет знак за фрейм
   - Зависит от шумовой «шерсти» голоса — хаотичен, легко вычисляется
   - Дополнительные 4 бита на фрейм

5. ВРЕМЕННА́Я НЕСТАБИЛЬНОСТЬ (jitter)
   - Δt между последовательными пиками давления — вибрации, дрожание голоса
   - Физиологически непредсказуем; период голосовых связок варьируется даже при попытке держать тон
"""

import time
import math
import hashlib
import threading
from collections import deque
from typing import List, Tuple, Optional

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠ sounddevice не установлен. Установите: pip install sounddevice")

# ─── Константы ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 44100      # Гц
FRAME_SIZE    = 2048       # Сэмплов на фрейм (~46 мс)
HOP_SIZE      = 512        # Шаг (~12 мс) — перекрытие фреймов
N_MFCC        = 13         # Число МФКЦ
N_FFT_BINS    = 16         # Голосовых FFT-бинов для энтропии
VOICE_LO_HZ   = 80        # Нижняя граница голоса
VOICE_HI_HZ   = 3400      # Верхняя граница голоса
MIN_FRAMES    = 20         # Минимум фреймов для генерации ключа
NOISE_FLOOR   = 0.002      # Порог активности (амплитуда 0–1)
H_MIN_BITS_PER_BYTE = 6.637
SAFETY_FACTOR = 3
MIN_POOL_BYTES_128 = 60
MIN_POOL_BYTES_256 = 117


def min_pool_bytes_for_key(key_bits: int) -> int:
    """Минимальный размер свежего пула по текущей SP800-90B калибровке."""
    if key_bits <= 128:
        return MIN_POOL_BYTES_128
    return MIN_POOL_BYTES_256


def is_microphone_available() -> bool:
    """
    Проверяет физическое наличие и доступность микрофона.
    Возвращает True только если sounddevice установлен И хотя бы одно
    устройство ввода реально доступно в системе.
    """
    if not SOUNDDEVICE_AVAILABLE:
        return False
    try:
        devices = sd.query_devices()
        for dev in devices:
            if dev["max_input_channels"] > 0:
                return True
        return False
    except Exception:
        return False


# ─── Вспомогательные DSP-функции ──────────────────────────────────────────────

def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 26,
                    fmin: float = 80.0, fmax: float = 3400.0) -> np.ndarray:
    """Создаёт треугольный мел-банк фильтров (n_mels × n_fft//2+1)."""
    def hz_to_mel(f): return 2595 * math.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)

    mel_lo = hz_to_mel(fmin)
    mel_hi = hz_to_mel(fmax)
    mel_points = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_points  = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        lo, mid, hi = bin_points[m-1], bin_points[m], bin_points[m+1]
        for k in range(lo, mid):
            if mid != lo:
                fb[m-1, k] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            if hi != mid:
                fb[m-1, k] = (hi - k) / (hi - mid)
    return fb


def _dct_matrix(n: int) -> np.ndarray:
    """Матрица DCT-II размера n×n (для МФКЦ из лог-мел-спектра)."""
    k = np.arange(n)
    m = np.arange(n)
    return np.cos(np.pi * m[:, None] * (2 * k[None, :] + 1) / (2 * n))


# Кеш мел-банка и DCT (зависит только от констант)
_FB  = _mel_filterbank(SAMPLE_RATE, FRAME_SIZE, n_mels=26)
_DCT = _dct_matrix(26)[:N_MFCC]          # только 13 строк


def compute_mfcc(frame: np.ndarray) -> np.ndarray:
    """
    Вычисляет N_MFCC МФКЦ для одного фрейма float32 [-1, 1].
    Возвращает массив float64 длины N_MFCC.
    """
    windowed  = frame * np.hanning(len(frame))
    spectrum  = np.abs(np.fft.rfft(windowed, n=FRAME_SIZE)) ** 2
    mel_power = _FB @ spectrum                          # (26,)
    log_mel   = np.log(mel_power + 1e-10)
    mfcc      = _DCT @ log_mel                          # (13,)
    return mfcc


def compute_spectral_entropy_bits(frame: np.ndarray) -> List[int]:
    """
    Извлекает энтропийные биты из FFT-фазы и амплитуды
    в голосовом диапазоне VOICE_LO_HZ…VOICE_HI_HZ.
    Возвращает список целых {0, 1}.
    """
    windowed = frame * np.hanning(len(frame))
    fft_full = np.fft.rfft(windowed, n=FRAME_SIZE)     # комплексный

    # Отображение частот на индексы бинов
    freqs    = np.fft.rfftfreq(FRAME_SIZE, 1 / SAMPLE_RATE)
    mask     = (freqs >= VOICE_LO_HZ) & (freqs <= VOICE_HI_HZ)
    voice_fft = fft_full[mask]

    # Берём ровно N_FFT_BINS бинов
    step  = max(1, len(voice_fft) // N_FFT_BINS)
    bins  = voice_fft[::step][:N_FFT_BINS]

    bits: List[int] = []

    for c in bins:
        # 3 младших бита целой части амплитуды × 1000
        amp = int(abs(c) * 1000) & 0xFF
        bits += [(amp >> i) & 1 for i in range(3)]

        # 3 младших бита фазы × 1000 (mod 8)
        phase_int = int((math.atan2(c.imag, c.real) + math.pi) / (2 * math.pi) * 1000) & 0xFF
        bits += [(phase_int >> i) & 1 for i in range(3)]

    return bits


def compute_zcr_bits(frame: np.ndarray) -> List[int]:
    """4 бита из нулевых пересечений (Zero-Crossing Rate)."""
    signs = np.sign(frame)
    zcr   = int(np.sum(np.abs(np.diff(signs))) / 2) % 16
    return [(zcr >> i) & 1 for i in range(4)]


def compute_jitter_bits(frame: np.ndarray) -> List[int]:
    """
    4 бита из временно́й нестабильности (jitter):
    разность соседних межпиковых интервалов.
    """
    # Находим локальные максимумы (пики давления)
    threshold = max(np.max(np.abs(frame)) * 0.3, NOISE_FLOOR)
    peaks = []
    for i in range(1, len(frame) - 1):
        if frame[i] > threshold and frame[i] > frame[i-1] and frame[i] > frame[i+1]:
            peaks.append(i)

    if len(peaks) < 3:
        return [0, 0, 0, 0]

    intervals = np.diff(peaks)                          # межпиковые интервалы
    jitter    = int(np.std(intervals) * 100) % 16
    return [(jitter >> i) & 1 for i in range(4)]


def compute_amplitude_bits(frame: np.ndarray) -> List[int]:
    """8 бит из мгновенных значений амплитуды (тепловой шум АЦП)."""
    # Берём 8 равноудалённых сэмплов, берём 1 младший бит каждого
    indices = np.linspace(0, len(frame) - 1, 8, dtype=int)
    raw     = (frame[indices] * 32767).astype(int)
    return [int(r) & 1 for r in raw]


def compute_mfcc_bits(mfcc: np.ndarray) -> List[int]:
    """
    12 бит из дробных частей МФКЦ.
    Дробная часть каждого коэффициента × 100 → берём 1 бит (чётность).
    """
    bits = []
    for coeff in mfcc[:N_MFCC-1]:                      # 12 коэффициентов
        frac = int(abs(coeff) * 100) % 2
        bits.append(frac)
    return bits


# ─── Основной класс ───────────────────────────────────────────────────────────

class VoiceEntropyCollector:
    """
    Сбор высокоэнтропийных данных голоса/звука.

    Использование:
        collector = VoiceEntropyCollector()
        collector.start()
        ...
        bits = collector.get_entropy_bits()
        collector.stop()
    """

    def __init__(self):
        self._lock            = threading.Lock()
        self._entropy_pool    : bytearray = bytearray()   # накопленные байты
        self._consumed_pool_bytes: int = 0
        self._bit_buffer      : List[int] = []            # текущий буфер битов
        self._frame_count     : int = 0
        self._active_frames   : int = 0                   # фреймы с голосом
        self._consumed_frame_count: int = 0
        self._consumed_active_frames: int = 0
        self._running              : bool = False
        self._microphone_confirmed : bool = False   # True только после успешного start()
        self._stream               : Optional[object] = None   # sd.InputStream
        self._sample_buffer   = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._buf_pos         : int = 0
        self._last_stats      = {}

        # История для статистики UI
        self._amplitude_history : deque = deque(maxlen=60)
        self._snr_history       : deque = deque(maxlen=30)
        self._entropy_quality   : float = 0.0

    # ── Управление потоком ────────────────────────────────────────────────────

    def start(self) -> bool:
        """Открывает поток микрофона. Возвращает True при успехе."""
        if not SOUNDDEVICE_AVAILABLE:
            return False
        if self._running:
            return True

        try:
            self._stream = sd.InputStream(
                samplerate  = SAMPLE_RATE,
                channels    = 1,
                dtype       = "float32",
                blocksize   = HOP_SIZE,
                callback    = self._audio_callback,
            )
            self._stream.start()
            self._running = True
            self._microphone_confirmed = True
            print("🎤 VoiceEntropyCollector запущен")
            return True
        except Exception as e:
            self._microphone_confirmed = False
            print(f"❌ Ошибка открытия микрофона: {e}")
            return False

    def stop(self):
        """Останавливает поток."""
        self._running = False
        self._microphone_confirmed = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        print("🎤 VoiceEntropyCollector остановлен")

    def reset(self):
        """Сбрасывает накопленные данные."""
        with self._lock:
            self._entropy_pool.clear()
            self._consumed_pool_bytes = 0
            self._bit_buffer.clear()
            self._frame_count  = 0
            self._active_frames = 0
            self._consumed_frame_count = 0
            self._consumed_active_frames = 0
            self._amplitude_history.clear()
            self._snr_history.clear()
            self._entropy_quality = 0.0

    # ── Callback микрофона ────────────────────────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status):
        """Вызывается sounddevice в аудио-потоке для каждого блока."""
        samples = indata[:, 0]                            # моно

        for s in samples:
            self._sample_buffer[self._buf_pos] = s
            self._buf_pos += 1

            if self._buf_pos >= FRAME_SIZE:
                self._process_frame(self._sample_buffer.copy())
                # Сдвигаем буфер на HOP_SIZE (перекрытие)
                self._sample_buffer[:FRAME_SIZE - HOP_SIZE] = \
                    self._sample_buffer[HOP_SIZE:]
                self._buf_pos = FRAME_SIZE - HOP_SIZE

    # ── Обработка фрейма ──────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray):
        """Извлекает биты энтропии из одного фрейма."""
        rms = float(np.sqrt(np.mean(frame ** 2)))
        self._amplitude_history.append(rms)

        self._frame_count += 1

        # Фреймы тишины дают минимальную энтропию — не отбрасываем,
        # но помечаем для статистики
        is_active = rms > NOISE_FLOOR
        if is_active:
            self._active_frames += 1

        # ── Сбор битов ────────────────────────────────────────────────────
        bits: List[int] = []

        # 1. Амплитудные биты (8 бит)
        bits.extend(compute_amplitude_bits(frame))

        # 2. ZCR биты (4 бита)
        bits.extend(compute_zcr_bits(frame))

        # 3. Jitter биты (4 бита)
        bits.extend(compute_jitter_bits(frame))

        # 4. Спектральные биты из FFT (N_FFT_BINS * 6 бит = 96 бит)
        bits.extend(compute_spectral_entropy_bits(frame))

        # 5. МФКЦ биты (12 бит) — только при активном голосе (тяжелее)
        if is_active:
            mfcc = compute_mfcc(frame)
            bits.extend(compute_mfcc_bits(mfcc))

        # 6. Субмикросекундный timestamp (4 бита)
        ts_bits = int((time.perf_counter() * 1e6)) & 0xF
        bits.extend([(ts_bits >> i) & 1 for i in range(4)])

        # ── Упаковка битов в байты и добавление в пул ─────────────────────
        with self._lock:
            self._bit_buffer.extend(bits)

            while len(self._bit_buffer) >= 8:
                byte_bits = self._bit_buffer[:8]
                del self._bit_buffer[:8]
                byte_val = sum(b << i for i, b in enumerate(byte_bits))
                self._entropy_pool.append(byte_val)

        # ── Обновление качества энтропии ──────────────────────────────────
        # voice_ratio: доля фреймов с реальным голосом (0.0–1.0)
        # pool_fill:   заполненность пула относительно 200 байт-цели
        # Итог: среднее двух метрик, выражается как 0.0–1.0
        if self._frame_count > 0:
            voice_ratio = self._active_frames / self._frame_count
            with self._lock:
                pool_fill = min(1.0, len(self._entropy_pool) / 200.0)
            self._entropy_quality = (voice_ratio + pool_fill) / 2.0


    # ── Получение данных ──────────────────────────────────────────────────────

    def get_entropy_bits(self) -> List[int]:
        """
        Возвращает все накопленные биты энтропии в виде списка {0, 1}.
        Не очищает пул.
        """
        with self._lock:
            byte_bits: List[int] = []
            for byte in self._entropy_pool:
                for i in range(8):
                    byte_bits.append((byte >> i) & 1)
            return byte_bits

    def get_entropy_bytes(self) -> bytes:
        """
        Возвращает копию байтового пула энтропии.
        Возвращает пустые байты если микрофон не был реально запущен —
        это исключает случайное использование пустого/фиктивного пула
        как источника энтропии при отсутствующем микрофоне.
        """
        if not self._microphone_confirmed:
            return b""
        with self._lock:
            return bytes(self._entropy_pool)

    def get_available_entropy_snapshot(self) -> dict:
        """
        Возвращает новые байты, ещё не использованные для генерации ключа,
        статистику свежего сегмента и конечные индексы снимка.
        Пул не очищается.
        """
        if not self._microphone_confirmed:
            return {
                "bytes": b"",
                "pool_end": self._consumed_pool_bytes,
                "frame_end": self._consumed_frame_count,
                "active_end": self._consumed_active_frames,
                "frame_count": 0,
                "active_frames": 0,
            }
        with self._lock:
            start = min(self._consumed_pool_bytes, len(self._entropy_pool))
            frame_count = self._frame_count - self._consumed_frame_count
            active_frames = self._active_frames - self._consumed_active_frames
            return {
                "bytes": bytes(self._entropy_pool[start:]),
                "pool_end": len(self._entropy_pool),
                "frame_end": self._frame_count,
                "active_end": self._active_frames,
                "frame_count": frame_count,
                "active_frames": active_frames,
            }

    def consume_entropy_until(self, pool_size: int,
                              frame_count: Optional[int] = None,
                              active_frames: Optional[int] = None):
        """Помечает байты до pool_size как использованные."""
        with self._lock:
            pool_size = max(0, min(pool_size, len(self._entropy_pool)))
            self._consumed_pool_bytes = max(self._consumed_pool_bytes, pool_size)
            if frame_count is not None:
                frame_count = max(0, min(frame_count, self._frame_count))
                self._consumed_frame_count = max(
                    self._consumed_frame_count, frame_count
                )
            if active_frames is not None:
                active_frames = max(0, min(active_frames, self._active_frames))
                self._consumed_active_frames = max(
                    self._consumed_active_frames, active_frames
                )

    def microphone_active(self) -> bool:
        """Возвращает True если микрофон был успешно открыт и работает."""
        return self._microphone_confirmed and self._running

    def has_enough_entropy(self, min_frames: int = MIN_FRAMES) -> bool:
        return self._frame_count >= min_frames

    def has_enough_voice_activity(self, max_silence_pct: float = 0.40) -> bool:
        """
        Возвращает True если доля тихих кадров не превышает max_silence_pct.

        silence_pct = (frame_count - active_frames) / frame_count

        Если silence_pct > max_silence_pct (по умолчанию 40%) —
        голос считается ненадёжным источником энтропии и не используется.
        Требует минимум MIN_FRAMES кадров для статистически значимой оценки.
        """
        if self._frame_count < MIN_FRAMES:
            return False
        silence_pct = (self._frame_count - self._active_frames) / self._frame_count
        return silence_pct <= max_silence_pct

    def silence_percent(self) -> float:
        """Возвращает долю тихих кадров от 0.0 до 1.0."""
        if self._frame_count == 0:
            return 1.0
        return (self._frame_count - self._active_frames) / self._frame_count

    def get_statistics(self) -> dict:
        """Статистика для обновления UI."""
        with self._lock:
            pool_size = len(self._entropy_pool)
            available_pool_size = max(0, pool_size - self._consumed_pool_bytes)
            available_frames = self._frame_count - self._consumed_frame_count
            available_active_frames = (
                self._active_frames - self._consumed_active_frames
            )

        rms_vals = list(self._amplitude_history)
        avg_rms  = float(np.mean(rms_vals)) if rms_vals else 0.0
        peak_rms = float(np.max(rms_vals))  if rms_vals else 0.0

        if available_frames > 0:
            fresh_voice_ratio = available_active_frames / available_frames
        else:
            fresh_voice_ratio = 0.0
        fresh_pool_fill = min(1.0, available_pool_size / 200.0)
        fresh_entropy_quality = (fresh_voice_ratio + fresh_pool_fill) / 2.0

        # dBFS
        def to_db(rms): return 20 * math.log10(rms + 1e-10)
        avg_db  = to_db(avg_rms)
        peak_db = to_db(peak_rms)

        # Приближённый SNR: отношение пикового RMS к floor
        signal_rms = max(rms_vals) if rms_vals else 0.0
        noise_rms  = min(rms_vals) if rms_vals else 1e-10
        snr_db = to_db(signal_rms / (noise_rms + 1e-10)) if noise_rms > 0 else 0.0

        # Основная частота (упрощённо: бин с макс. мощностью)
        dominant_freq = 0.0
        if rms_vals and avg_rms > NOISE_FLOOR:
            # Используем последний фрейм в буфере
            with self._lock:
                buf_snap = self._sample_buffer.copy()
            spectrum = np.abs(np.fft.rfft(buf_snap * np.hanning(FRAME_SIZE))) ** 2
            freqs    = np.fft.rfftfreq(FRAME_SIZE, 1 / SAMPLE_RATE)
            mask     = (freqs >= VOICE_LO_HZ) & (freqs <= VOICE_HI_HZ)
            if mask.any():
                dominant_freq = float(freqs[mask][np.argmax(spectrum[mask])])

        is_active = avg_rms > NOISE_FLOOR
        quality   = min(100.0, fresh_entropy_quality * 100)

        self._last_stats = {
            "is_active":     is_active,
            "frame_count":   self._frame_count,
            "active_frames": self._active_frames,
            "pool_bytes":    pool_size,
            "available_pool_bytes": available_pool_size,
            "available_frames": available_frames,
            "available_active_frames": available_active_frames,
            "avg_db":        avg_db,
            "peak_db":       peak_db,
            "snr_db":        snr_db,
            "dominant_freq": dominant_freq,
            "entropy_quality": quality,
            # Нормированные уровни для визуализатора [0–1]
            "audio_levels":  [min(1.0, r / 0.5) for r in rms_vals[-40:]],
        }
        return self._last_stats


# ─── Функции объединения голосовой и глазной энтропии ─────────────────────────

def combine_entropy(eye_bytes: bytes, voice_bytes: bytes) -> bytes:
    """
    Смешивает ВСЕ накопленные байты обоих источников в единый сырой пул.
    Хэширование и сжатие делегированы в derive_key / generate_combined_key.
    Конкатенация: eye_bytes || voice_bytes
    Сохраняем все биты обоих источников в оригинальном виде.
    Порядок фиксирован: сначала взгляд, потом голос.
    Возвращает bytes длиной len(eye) + len(voice).
    """
    if not eye_bytes and not voice_bytes:
        return b""
    pool = bytearray(eye_bytes) + bytearray(voice_bytes)

    return bytes(pool)


def derive_key(raw_pool: bytes, key_bits: int = 256,
               min_pool_bytes: Optional[int] = None) -> bytes:
    """
    Генерирует криптографический ключ из сырого пула энтропии
    единственным вызовом SHA-256.

    sha256_hash = SHA-256(raw_pool)   — 32 байта (256 бит)
    key         = sha256_hash[:key_bits // 8]

    Поддерживает key_bits = 128 (берём первые 16 байт) или 256 (все 32 байта).
    По умолчанию требует свежий пул не меньше текущей SP800-90B калибровки.
    """
    if key_bits not in (128, 256):
        return b""

    required_bytes = (
        min_pool_bytes
        if min_pool_bytes is not None
        else min_pool_bytes_for_key(key_bits)
    )
    if len(raw_pool) < required_bytes:
        return b""

    sha256_hash = hashlib.sha256(raw_pool).digest()   # единственный вызов хэша
    num_bytes   = key_bits // 8
    return sha256_hash[:num_bytes]


def generate_combined_key(eye_bytes: bytes,
                          voice_collector: VoiceEntropyCollector,
                          key_bits: int = 256) -> bytes:
    """
    Высокоуровневая функция: требует оба источника, объединяет свежие биты
    взгляда и голоса и возвращает готовый криптографический ключ.

    Поток данных:
        eye_bytes + voice_bytes
              ↓  combine_entropy()   — сырое смешивание, без хэширования
        raw_pool
              ↓  derive_key()        — всё хэширование здесь
        key (128 или 256 бит)

    При успешной генерации помечает использованные голосовые байты как
    израсходованные. Возвращает bytes длиной key_bits//8 или пустые байты.
    """
    voice_snapshot = voice_collector.get_available_entropy_snapshot()
    voice_bytes = voice_snapshot["bytes"]

    if not eye_bytes:
        print("⚠ Нет свежих данных взгляда для комбинированного режима")
        return b""
    if not voice_bytes:
        print("⚠ Нет свежих голосовых данных для комбинированного режима")
        return b""

    raw_pool = combine_entropy(eye_bytes, voice_bytes)
    min_bytes = min_pool_bytes_for_key(key_bits)
    if len(raw_pool) < min_bytes:
        print(f"⚠ Недостаточно свежего пула: {len(raw_pool)}/{min_bytes}B")
        return b""

    key = derive_key(raw_pool, key_bits)
    if key and voice_bytes:
        voice_collector.consume_entropy_until(
            voice_snapshot["pool_end"],
            voice_snapshot["frame_end"],
            voice_snapshot["active_end"],
        )
    print(f"✅ Ключ {key_bits} бит сгенерирован "
          f"(глаз: {len(eye_bytes)}B  голос: {len(voice_bytes)}B  "
          f"пул: {len(raw_pool)}B)")
    return key
