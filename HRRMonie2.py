import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import butter, sosfiltfilt, get_window

import uRAD_RP_SDK11

# =========================
# CONFIG FMCW (MANUEL OK)
# =========================
mode = 4  # FMCW Dual Rate
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
# BUFFERS
# =========================
buffer_phi = deque()
buffer_t = deque()

WINDOW_SECONDS = 30
MIN_SAMPLES = 200

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
# RANGE FFT (FMCW)
# =========================
def range_fft(I, Q):
    z = I + 1j*Q
    return np.fft.fft(z)

def select_bin(Z, prev_idx=None):
    amp = np.abs(Z)

    if prev_idx is None:
        return np.argmax(amp)

    # stabilisation légère
    local = amp[max(0, prev_idx-2):prev_idx+3]
    idx_local = np.argmax(local) + max(0, prev_idx-2)

    if amp[idx_local] > 0.9 * amp[prev_idx]:
        return idx_local

    return np.argmax(amp)

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
# HR SIMPLE
# =========================
def compute_hr(sig, fs):
    N = len(sig)
    X = np.fft.rfft(sig * get_window("hann", N))
    P = np.abs(X)**2
    f = np.fft.rfftfreq(N, 1/fs)

    mask = (f > 0.8) & (f < 2.0)
    f = f[mask]
    P = P[mask]

    return f[np.argmax(P)] * 60

# =========================
# PLOT HR
# =========================
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_ylim(40, 140)

times = []
hrs = []

# =========================
# MAIN LOOP
# =========================
idx_lock = None
fs_est = None
beta_fs = 0.05

while True:

    code, _, raw = uRAD_RP_SDK11.detection()
    if code != 0:
        closeProgram()

    I = np.array(raw[0])
    Q = np.array(raw[1])

    # =========================
    # DÉCOUPAGE FMCW MODE 4
    # =========================
    # 1ère rampe uniquement (Ns up)
    I_up = I[:Ns]
    Q_up = Q[:Ns]

    # =========================
    # FFT DISTANCE
    # =========================
    Z = range_fft(I_up, Q_up)
    idx_lock = select_bin(Z, idx_lock)

    z_sel = Z[idx_lock]

    # =========================
    # PHASE
    # =========================
    phi = unwrap(np.angle(z_sel))

    t_now = time.time()
    buffer_phi.append(phi)
    buffer_t.append(t_now)

    # fs estimation
    if len(buffer_t) > 1:
        dt = buffer_t[-1] - buffer_t[-2]
        fs_inst = 1.0 / max(dt, 1e-6)
        fs_est = fs_inst if fs_est is None else (1-beta_fs)*fs_est + beta_fs*fs_inst

    while buffer_t[-1] - buffer_t[0] > WINDOW_SECONDS:
        buffer_t.popleft()
        buffer_phi.popleft()

    if len(buffer_phi) < MIN_SAMPLES:
        continue

    phi_arr = np.array(buffer_phi)
    fs = fs_est

    # =========================
    # PIPELINE HR
    # =========================
    phi_hp = highpass(phi_arr, fs)
    phi_diff = np.diff(phi_hp, prepend=phi_hp[0])
    sig_hr = bandpass(phi_diff, 0.8, 2.0, fs)

    hr = compute_hr(sig_hr, fs)

    print(f"HR: {hr:.2f} bpm | bin: {idx_lock}")

    # =========================
    # GRAPH
    # =========================
    times.append(t_now)
    hrs.append(hr)

    xs = np.array(times) - times[0]
    line.set_data(xs, hrs)
    ax.set_xlim(max(0, xs[-1]-60), xs[-1]+1)

    fig.canvas.draw()
    plt.pause(0.01)