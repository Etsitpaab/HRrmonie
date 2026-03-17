import uRAD_RP_SDK11
import numpy as np
import scipy.signal as signal
import time

# =========================
# CONFIG
# =========================
FS = 20  # fréquence d'échantillonnage (Hz)

RESP_BAND = (0.1, 0.5)
HEART_BAND = (0.8, 2.0)

# =========================
# INIT uRAD
# =========================
uRAD_RP_SDK11.turnON()

# config radar (mode 2 FMCW recommandé)
mode = 2
f0 = 5
BW = 100
Ns = 200
Ntar = 1
Rmax = 10
MTI = 0
Mth = 0

uRAD_RP_SDK11.loadConfiguration(mode, f0, BW, Ns, Ntar, Rmax, MTI, Mth)

# =========================
# FILTRE
# =========================
def bandpass(data, low, high, fs):
    nyq = 0.5 * fs
    b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, data)

# =========================
# FFT
# =========================
def estimate_freq(sig):
    fft = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), 1/FS)

    mask = freqs > 0
    freqs = freqs[mask]
    fft = np.abs(fft[mask])

    return freqs[np.argmax(fft)]

# =========================
# LOOP
# =========================
while True:

    # acquisition radar
    ret, results, raw = uRAD_RP_SDK11.detection()

    if ret != 0:
        print("Erreur radar")
        continue

    I = np.array(raw[0])
    Q = np.array(raw[1])

    # =========================
    # PHASE
    # =========================
    phase = np.unwrap(np.arctan2(Q, I))
    phase = signal.detrend(phase)

    # =========================
    # FILTRAGE
    # =========================
    resp = bandpass(phase, RESP_BAND[0], RESP_BAND[1], FS)
    heart = bandpass(phase, HEART_BAND[0], HEART_BAND[1], FS)

    # =========================
    # FREQUENCES
    # =========================
    f_resp = estimate_freq(resp)
    f_heart = estimate_freq(heart)

    rr = f_resp * 60
    hr = f_heart * 60

    print(f"Respiration: {rr:.1f} BPM")
    print(f"Heart Rate: {hr:.1f} BPM")
    print("------")

    time.sleep(0.5)