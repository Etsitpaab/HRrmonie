import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import butter, sosfiltfilt, get_window

import uRAD_RP_SDK11

# =========================
# CONFIG RADAR FMCW
# =========================
mode = 4
f0 = 5
BW = 240
Ns = 200
Ntar = 1
Rmax = 100
MTI = 0
Mth = 0
Alpha = 10

distance_true = False
velocity_true = False
SNR_true = False
I_true = True
Q_true = True
movement_true = False

# =========================
# INIT RADAR
# =========================
def closeProgram():
    uRAD_RP_SDK11.turnOFF()
    exit()

if uRAD_RP_SDK11.turnON() != 0:
    closeProgram()

if uRAD_RP_SDK11.loadConfiguration(
    mode, f0, BW, Ns, Ntar, Rmax, MTI, Mth, Alpha,
    distance_true, velocity_true, SNR_true,
    I_true, Q_true, movement_true
) != 0:
    closeProgram()

# =========================
# PARAMÈTRES
# =========================
TARGET_FS = 20.0  # Hz

HR_MAX_STEP = 8.0
HR_ALPHA = 0.15
HR_MEDIAN_LEN = 5
HR_SNR_MIN = 2.0

# =========================
# BUFFERS
# =========================
buffer_phi = deque()
buffer_t = deque()
hr_buffer = deque(maxlen=HR_MEDIAN_LEN)

WINDOW_SECONDS = 30
MIN_SAMPLES = 200

hr_prev = None

# =========================
# FILTRES
# =========================
def bandpass(x, f1, f2, fs):
    nyq = fs / 2
    sos = butter(4, [f1/nyq, f2/nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)

def highpass(x, fs):
    nyq = fs / 2
    sos = butter(4, 0.05/nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, x)

# =========================
# RANGE FFT
# =========================
def range_fft(I, Q):
    return np.fft.fft(I + 1j*Q)

def select_bin(Z, prev_idx=None):
    amp = np.abs(Z)

    # suppression DC
    amp[:5] = 0

    if prev_idx is None:
        return np.argmax(amp)

    low = max(5, prev_idx - 3)
    high = min(len(amp), prev_idx + 4)

    idx_local = np.argmax(amp[low:high]) + low

    if amp[idx_local] > 0.8 * amp[prev_idx]:
        return idx_local

    return prev_idx

# =========================
# PHASE
# =========================
phase_prev = None
phase_unwrapped = 0

def unwrap(p):
    global phase_prev, phase_unwrapped

    if phase_prev is None:
        phase_prev = p
        phase_unwrapped = p
        return p

    dp = p - phase_prev

    if dp > np.pi:
        dp -= 2*np.pi
    elif dp < -np.pi:
        dp += 2*np.pi

    phase_unwrapped += dp
    phase_prev = p
    return phase_unwrapped

# =========================
# HR TRACKING
# =========================
def compute_hr_tracked(sig, fs):
    global hr_prev, hr_buffer

    N = len(sig)
    X = np.fft.rfft(sig * get_window("hann", N))
    P = np.abs(X)**2
    f = np.fft.rfftfreq(N, 1/fs)

    # bande cardiaque stricte
    mask = (f > 0.9) & (f < 1.8)
    f = f[mask]
    P = P[mask]

    idx = np.argmax(P)
    f_peak = f[idx]
    hr_raw = f_peak * 60

    noise = np.median(P)
    snr = P[idx] / (noise + 1e-9)

    if snr < HR_SNR_MIN:
        return hr_prev if hr_prev is not None else hr_raw

    if hr_prev is None:
        hr_prev = hr_raw
        hr_buffer.append(hr_raw)
        return hr_raw

    # correction douce si incohérent
    if abs(hr_raw - hr_prev) > 12:
        hr_raw = hr_prev + np.sign(hr_raw - hr_prev) * 6

    # limitation variation
    delta = hr_raw - hr_prev
    delta = np.clip(delta, -HR_MAX_STEP, HR_MAX_STEP)
    hr_new = hr_prev + delta

    # lissage
    hr_new = (1 - HR_ALPHA) * hr_prev + HR_ALPHA * hr_new

    hr_buffer.append(hr_new)
    hr_final = np.median(hr_buffer)

    hr_prev = hr_final

    return hr_final

# =========================
# PLOT
# =========================
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_ylim(40, 120)

times = []
hrs = []

# =========================
# MAIN LOOP
# =========================
idx_lock = None
last_time = time.time()

while True:

    # contrôle fréquence acquisition
    t_now = time.time()
    dt = t_now - last_time

    if dt < 1.0 / TARGET_FS:
        time.sleep((1.0 / TARGET_FS) - dt)
        t_now = time.time()

    last_time = t_now

    code, _, raw = uRAD_RP_SDK11.detection()
    if code != 0:
        closeProgram()

    I = np.array(raw[0])
    Q = np.array(raw[1])

    # FMCW : première rampe
    I_up = I[:Ns]
    Q_up = Q[:Ns]

    Z = range_fft(I_up, Q_up)
    idx_lock = select_bin(Z, idx_lock)

    z_sel = Z[idx_lock]

    phi = unwrap(np.angle(z_sel))

    buffer_phi.append(phi)
    buffer_t.append(t_now)

    while buffer_t[-1] - buffer_t[0] > WINDOW_SECONDS:
        buffer_t.popleft()
        buffer_phi.popleft()

    if len(buffer_phi) < MIN_SAMPLES:
        continue

    phi_arr = np.array(buffer_phi)
    fs = TARGET_FS

    phi_hp = highpass(phi_arr, fs)
    phi_diff = np.diff(phi_hp, prepend=phi_hp[0])
    sig_hr = bandpass(phi_diff, 0.9, 1.8, fs)

    hr = compute_hr_tracked(sig_hr, fs)

    print(f"HR_raw tracking -> {hr:.2f} bpm | bin: {idx_lock}")

    times.append(t_now)
    hrs.append(hr)

    xs = np.array(times) - times[0]
    line.set_data(xs, hrs)
    ax.set_xlim(max(0, xs[-1]-60), xs[-1]+1)

    fig.canvas.draw()
    plt.pause(0.01)