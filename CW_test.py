import time
import numpy as np
import uRAD_RP_SDK11

# =========================
# Configuration uRAD
# =========================
mode = 3        # FMCW triangular
f0 = 5          # 24.005 GHz
BW = 240        # MHz
Ns = 200        # samples per chirp (raw = 2*Ns)
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

def closeProgram():
    try:
        uRAD_RP_SDK11.turnOFF()
    except Exception:
        pass
    raise SystemExit

if uRAD_RP_SDK11.turnON() != 0:
    closeProgram()

if uRAD_RP_SDK11.loadConfiguration(
    mode, f0, BW, Ns, Ntar, Rmax, MTI, Mth, Alpha,
    distance_true, velocity_true, SNR_true,
    I_true, Q_true, movement_true
) != 0:
    closeProgram()

# =========================
# Paramètres traitement
# =========================
N_FFT = 1024
window = np.hanning(Ns)

# Axe distance (formule constructeur)
Dmax = 75.0 * Ns / BW
dist_axis = np.linspace(0, Dmax, N_FFT // 2, endpoint=False)

k_min = 2  # éviter DC
ROI_MIN, ROI_MAX = 0.30, 3.50  # zone humaine typique (à adapter)
roi_mask = (dist_axis >= ROI_MIN) & (dist_axis <= ROI_MAX)

# --- Clutter map (suppression statique range-domain) ---
clutter = np.zeros(N_FFT // 2, dtype=np.float64)
beta_clutter = 0.02  # plus petit = plus "statique"

# --- Tracking distance (gating + lissage) ---
R_track = None
alpha_track = 0.15       # filtre IIR sur la distance
gate_m = 0.8             # recherche pic dans ± gate_m autour de R_track quand lock
lock_hold_s = 1.0        # tolérance de perte avant reset
last_lock_time = None

# --- CFAR 1D ---
GUARD = 4
TRAIN = 16
SCALE = 6.0  # seuil (augmente si trop de faux pics)

def ca_cfar_1d(power, guard=4, train=16, scale=6.0):
    n = len(power)
    det = np.zeros(n, dtype=bool)
    thr = np.zeros(n, dtype=np.float64)
    start = train + guard
    end = n - (train + guard)
    for k in range(start, end):
        left = power[k - guard - train : k - guard]
        right = power[k + guard + 1 : k + guard + train + 1]
        noise = (np.sum(left) + np.sum(right)) / (2 * train + 1e-12)
        t = noise * scale
        thr[k] = t
        det[k] = power[k] > t
    return det, thr

# =========================
# FIR bandpass (fenêtre Hann) pour vital signs
# =========================
def fir_bandpass(fs, f1, f2, numtaps=401):
    """
    Filtre FIR passe-bande par sinc fenêtrée.
    fs en Hz, f1<f2 en Hz.
    """
    f1 = max(f1, 1e-6)
    f2 = min(f2, 0.499 * fs)
    n = np.arange(numtaps) - (numtaps - 1) / 2
    h2 = 2 * f2 / fs * np.sinc(2 * f2 / fs * n)
    h1 = 2 * f1 / fs * np.sinc(2 * f1 / fs * n)
    h = h2 - h1
    w = np.hanning(numtaps)
    h *= w
    h /= np.sum(h + 1e-12)  # normalisation grossière (DC pas critique ici)
    return h

def apply_fir(x, h):
    # convolution "same"
    return np.convolve(x, h, mode="same")

def dominant_freq_hz(x, fs, fmin, fmax):
    """
    Pic spectral dans [fmin, fmax] via FFT.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if len(x) < 16:
        return None

    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), d=1.0/fs)
    P = np.abs(X)**2

    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return None

    k = np.argmax(P[band])
    return f[band][k]

# =========================
# Buffers vital signs
# =========================
phase_buf = []
t_buf = []

# Fenêtres d'analyse (tu peux ajuster)
WIN_S = 30.0   # secondes (resp OK à 20-30s, coeur mieux 30-45s)
MIN_S = 15.0   # ne pas estimer avant

# =========================
# Boucle acquisition
# =========================
t0 = time.time()
t_prev = None

print("Démarrage... Ctrl+C pour arrêter.")

try:
    while True:
        rc, results, raw = uRAD_RP_SDK11.detection()
        if rc != 0:
            closeProgram()

        now = time.time()
        if t_prev is None:
            t_prev = now
        dt = now - t_prev
        t_prev = now

        # Récup I/Q
        I = np.asarray(raw[0], dtype=np.float64)
        Q = np.asarray(raw[1], dtype=np.float64)

        # Chirp UP uniquement
        I_up = I[:Ns]
        Q_up = Q[:Ns]
        s_up = (I_up + 1j * Q_up)

        # Preproc
        s_up = s_up - np.mean(s_up)     # DC removal
        s_up = s_up * window

        # Range FFT
        S = np.fft.fft(s_up, n=N_FFT)[:N_FFT // 2]
        mag = np.abs(S)

        # Clutter suppression (MTI range-domain)
        clutter = (1 - beta_clutter) * clutter + beta_clutter * mag
        mag_mti = mag - clutter
        mag_mti[mag_mti < 0] = 0.0

        power = mag_mti**2

        # ROI humaine
        power_roi = power.copy()
        power_roi[~roi_mask] = 0.0

        # Gate autour du tracking si lock
        if R_track is not None:
            gate = (dist_axis >= (R_track - gate_m)) & (dist_axis <= (R_track + gate_m))
            power_roi = power_roi * gate

        # CFAR
        det, thr = ca_cfar_1d(power_roi, guard=GUARD, train=TRAIN, scale=SCALE)
        cand = np.where(det)[0]
        cand = cand[cand >= k_min]

        if cand.size == 0:
            # gestion perte lock
            if last_lock_time is not None and (now - last_lock_time) > lock_hold_s:
                R_track = None
                last_lock_time = None
                # reset buffers vital si on perd la cible
                phase_buf.clear()
                t_buf.clear()
            print("No reliable detection (CFAR).")
            continue

        # Choix du pic : ici on prend le plus fort dans ROI (+ gate)
        k_peak = cand[np.argmax(power_roi[cand])]
        R_meas = dist_axis[k_peak]

        # Tracking
        if R_track is None:
            R_track = R_meas
        else:
            R_track = (1 - alpha_track) * R_track + alpha_track * R_meas

        last_lock_time = now

        # =========================
        # Vital signs: phase au bin suivi
        # =========================
        # On prend la valeur complexe S[k_peak] (chirp UP), phase -> micro-mouvements
        z = S[k_peak]
        phi = np.angle(z)

        # Buffer temporel
        phase_buf.append(phi)
        t_buf.append(now)

        # Garder ~ WIN_S secondes
        while (t_buf[-1] - t_buf[0]) > WIN_S:
            phase_buf.pop(0)
            t_buf.pop(0)

        # Estimation FS (fréq d'échantillonnage) via timestamps
        duration = t_buf[-1] - t_buf[0]
        n = len(t_buf)
        fs = (n - 1) / duration if duration > 1e-6 else None

        # Unwrap phase + dérive lente (detrend simple)
        resp_bpm = None
        hr_bpm = None

        if fs is not None and duration >= MIN_S and n > 64:
            ph = np.unwrap(np.array(phase_buf, dtype=np.float64))
            ph = ph - np.mean(ph)

            # Detrend très simple : enlever une droite (évite drift)
            tt = np.array(t_buf, dtype=np.float64)
            tt = tt - tt[0]
            A = np.vstack([tt, np.ones_like(tt)]).T
            m, c = np.linalg.lstsq(A, ph, rcond=None)[0]
            ph_d = ph - (m * tt + c)

            # Respiration: ~0.1 à 0.6 Hz (6 à 36 bpm)
            h_resp = fir_bandpass(fs, 0.10, 0.60, numtaps=301)
            resp_sig = apply_fir(ph_d, h_resp)
            f_resp = dominant_freq_hz(resp_sig, fs, 0.10, 0.60)
            if f_resp is not None:
                resp_bpm = 60.0 * f_resp

            # Coeur: ~0.8 à 3.0 Hz (48 à 180 bpm)
            # (souvent faible -> fenêtre plus longue aide)
            h_hr = fir_bandpass(fs, 0.80, 3.00, numtaps=401)
            hr_sig = apply_fir(ph_d, h_hr)
            f_hr = dominant_freq_hz(hr_sig, fs, 0.80, 3.00)
            if f_hr is not None:
                hr_bpm = 60.0 * f_hr

        # Affichage
        if resp_bpm is None and hr_bpm is None:
            print(f"R_track={R_track:.2f} m | (collecting) buf={len(phase_buf)} dur={duration:.1f}s")
        else:
            print(f"R_track={R_track:.2f} m | Resp={resp_bpm:.1f} bpm | HR={hr_bpm:.1f} bpm | fs~{fs:.1f} Hz")

except KeyboardInterrupt:
    closeProgram()