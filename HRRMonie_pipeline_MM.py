# =========================
# IMPORTS
# =========================
import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, get_window
import uRAD_RP_SDK11
import matplotlib.pyplot as plt
from collections import deque

# =========================
# TRACKING ROBUSTE
# =========================
class PeakTracker:
    def __init__(self, max_jump_hz=0.15):
        self.prev_freq = None
        self.max_jump = max_jump_hz

    def update(self, freqs, Pxx, fmin, fmax):
        mask = (freqs >= fmin) & (freqs <= fmax)
        f = freqs[mask]
        p = Pxx[mask]

        if len(p) == 0:
            return np.nan

        peaks, _ = find_peaks(p, prominence=0.1 * np.max(p))
        if len(peaks) == 0:
            return np.nan

        candidates = f[peaks]

        if self.prev_freq is not None:
            dist = np.abs(candidates - self.prev_freq)
            idx = np.argmin(dist)

            if dist[idx] > self.max_jump:
                return np.nan

            chosen = candidates[idx]
        else:
            chosen = candidates[np.argmax(p[peaks])]

        self.prev_freq = chosen
        return chosen


# =========================
# KALMAN HR
# =========================
class KalmanHR:
    def __init__(self, dt=1.0):
        self.x = np.array([1.5, 0.0])
        self.P = np.eye(2)

        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])

        self.Q = np.array([[1e-3, 0], [0, 1e-2]])
        self.R_base = 0.05

    def update(self, z, snr):
        if not np.isfinite(z):
            return self.x[0]

        R = self.R_base / max(snr, 1)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return self.x[0]


# =========================
# SMOOTHING ADAPTATIF
# =========================
class AdaptiveSmoother:
    def __init__(self):
        self.value = None

    def update(self, x, snr):
        if not np.isfinite(x):
            return self.value

        alpha = np.clip(snr / 10.0, 0.05, 0.5)

        if self.value is None:
            self.value = x
        else:
            self.value = (1 - alpha) * self.value + alpha * x

        return self.value


# =========================
# PARAMS
# =========================
HR_MIN_HZ = 0.90
HR_MAX_HZ = 2.00

TARGET_SMOOTH_ALPHA = 0.15
TARGET_NEIGHBOR_MARGIN = 1

# =========================
# INIT TRACKING
# =========================
peak_tracker = PeakTracker()
kalman_hr = KalmanHR()
smooth_hr = AdaptiveSmoother()

# =========================
# RADAR INIT
# =========================
uRAD_RP_SDK11.turnON()
uRAD_RP_SDK11.loadConfiguration(1, 125, 240, 200, 3, 100, 0, 0, 10,
                                False, False, False, True, True, False)

# =========================
# BIN TRACKING
# =========================
score_bins = None
idx_lock = None

def init_score_bins(n):
    global score_bins
    score_bins = np.zeros(n)

def choisir_idx_stable(z_bin, idx_prev):
    global score_bins

    amp = np.abs(z_bin)
    score_bins = (1 - TARGET_SMOOTH_ALPHA) * score_bins + TARGET_SMOOTH_ALPHA * amp

    left = max(0, idx_prev - 2)
    right = min(len(score_bins), idx_prev + 3)

    dist_penalty = np.abs(np.arange(left, right) - idx_prev)
    score_local = score_bins[left:right] - 0.2 * dist_penalty

    return left + int(np.argmax(score_local))


# =========================
# SIGNAL BUFFER
# =========================
buffer = deque()
time_buffer = deque()

# =========================
# MAIN LOOP
# =========================
while True:
    _, _, iq = uRAD_RP_SDK11.detection()

    I = np.array(iq[0])
    Q = np.array(iq[1])
    z = I + 1j * Q

    if idx_lock is None:
        init_score_bins(len(z))
        idx_lock = int(np.argmax(np.abs(z)))
    else:
        idx_lock = choisir_idx_stable(z, idx_lock)

    z_sel = np.mean(z[max(0, idx_lock-1):idx_lock+2])

    phi = np.angle(z_sel)

    t = time.time()
    buffer.append(phi)
    time_buffer.append(t)

    if len(buffer) < 200:
        continue

    # FFT
    sig = np.array(buffer)
    fs = 1 / np.median(np.diff(time_buffer))

    N = len(sig)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    Pxx = np.abs(np.fft.rfft(sig))**2

    # Peak tracking
    f_peak = peak_tracker.update(freqs, Pxx, HR_MIN_HZ, HR_MAX_HZ)

    # SNR approx
    noise = np.median(Pxx)
    signal = np.max(Pxx)
    snr = 10 * np.log10(signal / (noise + 1e-9))

    # Kalman
    f_kalman = kalman_hr.update(f_peak, snr)
    hr_kalman = f_kalman * 60

    # Adaptive smoothing
    hr_final = smooth_hr.update(hr_kalman, snr)

    print(f"HR: {hr_final:.2f} bpm | raw: {f_peak*60 if np.isfinite(f_peak) else np.nan:.2f} | SNR: {snr:.2f} | bin: {idx_lock}")