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

# ROI humaine (à adapter à ton usage)
ROI_MIN, ROI_MAX = 0.30, 3.50
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

# --- Sélection cible (anti multipath dominant) ---
w_d = 3.0  # pénalité distance dans le score (augmente si ça saute vers un écho lointain)
min_rel_peak = 0.30  # au premier lock, la détection doit être >= 30% du max candidat pour être considérée "robuste"

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
    f1 = max(f1, 1e-6)
    f2 = min(f2, 0.499 * fs)
    n = np.arange(numtaps) - (numtaps - 1) / 2
    h2 = 2 * f2 / fs * np.sinc(2 * f2 / fs * n)
    h1 = 2 * f1 / fs * np.sinc(2 * f1 / fs * n)
    h = h2 - h1
    w = np.hanning(numtaps)
    h *= w
    # normalisation énergie approximative
    h /= (np.sum(np.abs(h)) + 1e-12)
    return h

def apply_fir(x, h):
    return np.convolve(x, h, mode="same")

def dominant_freq_hz(x, fs, fmin, fmax):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if len(x) < 32:
        return None
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    P = np.abs(X) ** 2
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return None
    k = np.argmax(P[band])
    return float(f[band][k])

# =========================
# Buffers vital signs
# =========================
phase_buf = []
t_buf = []

WIN_S = 30.0   # secondes (augmente à 45-60s pour améliorer HR)
MIN_S = 15.0

print("Démarrage... Ctrl+C pour arrêter.")

try:
    t_prev = None
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
        s_up = (I[:Ns] + 1j * Q[:Ns])

        # Preproc
        s_up = s_up - np.mean(s_up)
        s_up = s_up * window

        # Range FFT
        S = np.fft.fft(s_up, n=N_FFT)[:N_FFT // 2]
        mag = np.abs(S)

        # Clutter suppression (MTI range-domain)
        clutter = (1 - beta_clutter) * clutter + beta_clutter * mag
        mag_mti = mag - clutter
        mag_mti[mag_mti < 0] = 0.0

        power = mag_mti ** 2

        # ROI humaine
        power_roi = power.copy()
        power_roi[~roi_mask] = 0.0

        # Gate autour du tracking si lock
        if R_track is not None:
            gate = (dist_axis >= (R_track - gate_m)) & (dist_axis <= (R_track + gate_m))
            power_roi *= gate

        # CFAR
        det, thr = ca_cfar_1d(power_roi, guard=GUARD, train=TRAIN, scale=SCALE)
        cand = np.where(det)[0]
        cand = cand[cand >= k_min]

        if cand.size == 0:
            # gestion perte lock
            if last_lock_time is not None and (now - last_lock_time) > lock_hold_s:
                R_track = None
                last_lock_time = None
                phase_buf.clear()
                t_buf.clear()
            print("No reliable detection (CFAR).")
            continue

        # =========================
        # Choix du bin cible (correction anti multipath)
        # =========================
        if R_track is None:
            # Premier lock: privilégier la cible la plus proche mais "robuste"
            cand_sorted = cand[np.argsort(dist_axis[cand])]
            pmax = np.max(power_roi[cand_sorted])
            good = cand_sorted[power_roi[cand_sorted] > (min_rel_peak * pmax)]
            k_peak = int(good[0] if good.size else cand_sorted[0])
            R_meas = float(dist_axis[k_peak])
            R_track = R_meas
        else:
            # Lock existant: score puissance - pénalité distance à R_track
            d = np.abs(dist_axis[cand] - R_track)
            p = power_roi[cand] / (np.max(power_roi[cand]) + 1e-12)
            score = p - w_d * (d / (gate_m + 1e-6))
            k_peak = int(cand[np.argmax(score)])
            R_meas = float(dist_axis[k_peak])
            R_track = (1 - alpha_track) * R_track + alpha_track * R_meas

        last_lock_time = now

        # =========================
        # Vital signs: phase au bin suivi
        # =========================
        z = S[k_peak]
        phi = float(np.angle(z))

        phase_buf.append(phi)
        t_buf.append(now)

        # garder WIN_S secondes
        while (t_buf[-1] - t_buf[0]) > WIN_S:
            phase_buf.pop(0)
            t_buf.pop(0)

        duration = t_buf[-1] - t_buf[0]
        n = len(t_buf)
        fs = (n - 1) / duration if duration > 1e-6 else None

        resp_rpm = None   # respiration rate per minute
        hr_bpm = None     # heart rate beats per minute

        if fs is not None and duration >= MIN_S and n > 64:
            ph = np.unwrap(np.array(phase_buf, dtype=np.float64))
            ph -= np.mean(ph)

            # detrend linéaire (drift)
            tt = np.array(t_buf, dtype=np.float64)
            tt -= tt[0]
            A = np.vstack([tt, np.ones_like(tt)]).T
            m, c = np.linalg.lstsq(A, ph, rcond=None)[0]
            ph_d = ph - (m * tt + c)

            # Respiration: 0.10–0.60 Hz => 6–36 rpm
            h_resp = fir_bandpass(fs, 0.10, 0.60, numtaps=301)
            resp_sig = apply_fir(ph_d, h_resp)
            f_resp = dominant_freq_hz(resp_sig, fs, 0.10, 0.60)
            if f_resp is not None:
                resp_rpm = 60.0 * f_resp

            # Heart: 0.80–3.00 Hz => 48–180 bpm
            h_hr = fir_bandpass(fs, 0.80, 3.00, numtaps=401)
            hr_sig = apply_fir(ph_d, h_hr)
            f_hr = dominant_freq_hz(hr_sig, fs, 0.80, 3.00)
            if f_hr is not None:
                hr_bpm = 60.0 * f_hr

        # =========================
        # Affichage
        # =========================
        peak_db = 10.0 * np.log10(power_roi[k_peak] + 1e-12)

        if resp_rpm is None and hr_bpm is None:
            print(f"R_track={R_track:.2f} m | peak={peak_db:.1f} dB | collecting buf={n} dur={duration:.1f}s")
        else:
            # Respiration en rpm (respirations/min), coeur en bpm
            print(
                f"R_track={R_track:.2f} m | Resp={resp_rpm:.1f} rpm | HR={hr_bpm:.1f} bpm | fs~{fs:.1f} Hz | peak={peak_db:.1f} dB"
            )

except KeyboardInterrupt:
    closeProgram()