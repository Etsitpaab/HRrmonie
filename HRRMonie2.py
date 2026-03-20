import time
import csv
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks, get_window
import uRAD_RP_SDK11

# =========================
# Configuration radar uRAD
# =========================
mode = 1  # CW Doppler
f0 = 125
BW = 240
Ns = 200
Ntar = 3
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
# Paramètres physiologiques
# =========================
HR_MIN_HZ = 0.80          # 48 bpm
HR_MAX_HZ = 2.50          # 150 bpm
HR_HALF_BAND_HZ = 0.35
HR_HARD_JUMP_HZ = 0.16
HR_ALPHA_TRACK = 0.18
HR_SNR_MIN = 2.5
HR_PROM_MIN = 0.015
HR_MEDIAN_LEN = 5
HR_EMA_ALPHA_MIN = 0.08
HR_EMA_ALPHA_MAX = 0.32
HR_MAX_STEP_BPM = 2.5     # limite d’évolution par sortie

RR_MIN_HZ = 0.10
RR_MAX_HZ = 0.50

# =========================
# Tracking du bin (léger)
# =========================
TARGET_SMOOTH_ALPHA = 0.15
TARGET_REACQ_PERIOD = 120
TARGET_NEIGHBOR_MARGIN = 0   # bin unique pour préserver le signal utile
TARGET_LOCAL_RADIUS = 2
TARGET_SWITCH_RATIO = 1.12
TARGET_HYST_BINS = 4

# =========================
# Motion gating (proposition du pipeline)
# =========================
MOTION_GATE_SECONDS = 6.0
MOTION_VAR_THR = 0.18
MOTION_DIFF_MAD_THR = 0.12
MOTION_FREEZE_ALPHA = 0.05  # si fenêtre instable, on continue de sortir une valeur mais on freine fort

# =========================
# Affichage / buffers
# =========================
WINDOW_SECONDS = 30.0
MIN_SAMPLES = 200
PRINT_PERIOD = 1.0
SHOW_HR_GRAPH = True

# =========================
# Logging
# =========================
ENABLE_LOG = True
LOG_DIR = Path("logs_hr")

# =========================
# États globaux
# =========================
epsilon = 1e-6
phase_deplie = 0.0
phase_precedente = None
fs_est = None
beta_fs = 0.05

# I/Q display / correction
beta_dc_iq = 0.001
beta_variance_iq = 0.001
moyenne_I_iq = None
moyenne_Q_iq = None
variance_I_iq = None
variance_Q_iq = None

# Phase preprocessing: retrait DC + correction légère de déséquilibre I/Q
beta_dc_phase = 0.01
beta_var_phase = 0.01
moyenne_I_phase = None
moyenne_Q_phase = None
variance_I_phase = None
variance_Q_phase = None

idx_lock = None
reacq_count = 0
score_bins = None

buffer_temps = deque()
buffer_phi = deque()

hr_history = deque(maxlen=HR_MEDIAN_LEN)
hr_last_stable_hz = np.nan
hr_last_output = np.nan
rr_last_output = np.nan

marqueur_print = time.time()

# séries pour graphe HR
plot_times = deque(maxlen=240)
plot_hr = deque(maxlen=240)
plot_hr_raw = deque(maxlen=240)

# logging
log_writer = None
log_file_handle = None
log_file_path = None


# =========================
# Utilitaires radar
# =========================
def closeProgram():
    try:
        if log_file_handle is not None:
            log_file_handle.close()
    except Exception:
        pass
    uRAD_RP_SDK11.turnOFF()
    raise SystemExit


def init_radar():
    return_code = uRAD_RP_SDK11.turnON()
    if return_code != 0:
        closeProgram()

    return_code = uRAD_RP_SDK11.loadConfiguration(
        mode, f0, BW, Ns, Ntar, Rmax, MTI, Mth, Alpha,
        distance_true, velocity_true, SNR_true,
        I_true, Q_true, movement_true,
    )
    if return_code != 0:
        closeProgram()


# =========================
# Logging
# =========================
def init_logger():
    global log_writer, log_file_handle, log_file_path
    if not ENABLE_LOG:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file_path = LOG_DIR / f"vitals_log_{ts}.csv"
    log_file_handle = open(log_file_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file_handle)
    log_writer.writerow([
        "timestamp", "fs_est", "idx_lock", "motion_score", "motion_ok",
        "rr_bpm", "rr_hz", "rr_snr", "rr_prom",
        "hr_bpm", "hr_raw_bpm", "hr_med_bpm", "hr_hz_ref",
        "hr_snr", "hr_prom", "alpha", "max_step_bpm"
    ])
    log_file_handle.flush()


def write_log(row):
    if log_writer is None:
        return
    log_writer.writerow(row)
    log_file_handle.flush()


# =========================
# Prétraitement I/Q
# =========================
def pretraitement_affichage_iq(I_entree, Q_entree):
    global moyenne_I_iq, moyenne_Q_iq, variance_I_iq, variance_Q_iq

    I_entree = float(I_entree)
    Q_entree = float(Q_entree)

    if moyenne_I_iq is None:
        moyenne_I_iq, moyenne_Q_iq = I_entree, Q_entree
        variance_I_iq, variance_Q_iq = 1.0, 1.0

    moyenne_I_iq = (1 - beta_dc_iq) * moyenne_I_iq + beta_dc_iq * I_entree
    moyenne_Q_iq = (1 - beta_dc_iq) * moyenne_Q_iq + beta_dc_iq * Q_entree

    I_centre = I_entree - moyenne_I_iq
    Q_centre = Q_entree - moyenne_Q_iq

    variance_I_iq = (1 - beta_variance_iq) * variance_I_iq + beta_variance_iq * (I_centre ** 2)
    variance_Q_iq = (1 - beta_variance_iq) * variance_Q_iq + beta_variance_iq * (Q_centre ** 2)

    I_normalise = I_centre / np.sqrt(variance_I_iq + epsilon)
    Q_normalise = Q_centre / np.sqrt(variance_Q_iq + epsilon)
    return I_normalise, Q_normalise


def pretraitement_phase(I_entree, Q_entree):
    """
    Pipeline du PDF : retrait DC + correction légère du déséquilibre I/Q avant atan2.
    On centre I et Q puis on les normalise par leur variance glissante.
    """
    global moyenne_I_phase, moyenne_Q_phase, variance_I_phase, variance_Q_phase

    I_entree = float(I_entree)
    Q_entree = float(Q_entree)

    if moyenne_I_phase is None:
        moyenne_I_phase = I_entree
        moyenne_Q_phase = Q_entree
        variance_I_phase = 1.0
        variance_Q_phase = 1.0

    moyenne_I_phase = (1 - beta_dc_phase) * moyenne_I_phase + beta_dc_phase * I_entree
    moyenne_Q_phase = (1 - beta_dc_phase) * moyenne_Q_phase + beta_dc_phase * Q_entree

    I_centre = I_entree - moyenne_I_phase
    Q_centre = Q_entree - moyenne_Q_phase

    variance_I_phase = (1 - beta_var_phase) * variance_I_phase + beta_var_phase * (I_centre ** 2)
    variance_Q_phase = (1 - beta_var_phase) * variance_Q_phase + beta_var_phase * (Q_centre ** 2)

    I_eq = I_centre / np.sqrt(variance_I_phase + epsilon)
    Q_eq = Q_centre / np.sqrt(variance_Q_phase + epsilon)
    return I_eq, Q_eq


# =========================
# Dépliage de phase
# =========================
def unwrap_phase(phase_actuelle):
    global phase_deplie, phase_precedente

    if phase_precedente is None:
        phase_precedente = phase_actuelle
        phase_deplie = phase_actuelle
        return phase_deplie

    delta = phase_actuelle - phase_precedente
    if delta > np.pi:
        delta -= 2 * np.pi
    elif delta < -np.pi:
        delta += 2 * np.pi

    phase_deplie += delta
    phase_precedente = phase_actuelle
    return phase_deplie


# =========================
# Filtres
# =========================
def filtre_pass_haut(signal, fs, fc=0.05):
    signal = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0
    if fc >= nyq:
        return signal.copy()
    sos = butter(4, fc / nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, signal)


def passe_bande(signal, f_basse, f_haut, fs, ordre=4):
    signal = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0
    if f_basse <= 0 or f_haut >= nyq or f_basse >= f_haut:
        raise ValueError("Bornes de passe-bande invalides.")
    sos = butter(ordre, [f_basse / nyq, f_haut / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def notch_resp_harmonics(signal, fs, f_rr, max_harm=6, bw=0.06):
    x = np.asarray(signal, dtype=np.float64).copy()
    if not np.isfinite(f_rr) or f_rr <= 0:
        return x

    nyq = fs / 2.0
    for k in range(2, max_harm + 1):
        f0 = k * f_rr
        if f0 - bw <= 0:
            continue
        if f0 + bw >= nyq:
            break
        low = max(0.01, f0 - bw)
        high = min(nyq - 1e-3, f0 + bw)
        try:
            band = passe_bande(x, low, high, fs, ordre=2)
            x = x - band
        except ValueError:
            pass
    return x


# =========================
# Analyse spectrale
# =========================
def puissance_spectre(x, fs, nfft=None, window="hann"):
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(max(N, 1))))
    w = get_window(window, N)
    xw = (x - np.mean(x)) * w
    X = np.fft.rfft(xw, n=nfft)
    Pxx = np.abs(X) ** 2
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return f, Pxx


def qualite_pics(frequence, Pxx, fmin, fmax):
    bande = (frequence >= fmin) & (frequence <= fmax)
    frequence_bande = frequence[bande]
    Pxx_bande = Pxx[bande]

    if Pxx_bande.size == 0:
        return np.nan, np.nan, 0.0

    prom_min = 0.08 * np.max(Pxx_bande)
    pics, propriete = find_peaks(Pxx_bande, prominence=prom_min)
    bruit = np.median(Pxx_bande) + 1e-12

    if len(pics) == 0:
        k = int(np.argmax(Pxx_bande))
        frequence_pic = frequence_bande[k]
        pic_pow = Pxx_bande[k] + 1e-12
        SNR_db = 10.0 * np.log10(pic_pow / bruit)
        return frequence_pic, SNR_db, 0.0

    idx_best = np.argmax(Pxx_bande[pics])
    k0 = pics[idx_best]
    frequence_pic = frequence_bande[k0]
    pic_pow = Pxx_bande[k0] + 1e-12
    SNR_db = 10.0 * np.log10(pic_pow / bruit)
    prominences = propriete["prominences"]
    prom = float(prominences[idx_best])
    prominence_norm = prom / pic_pow
    return frequence_pic, SNR_db, prominence_norm


def FFT_glissante(x, fs, freq_min, freq_max, duree_fenetre=20.0):
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    fen = int(round(duree_fenetre * fs))
    if fen < 5:
        raise ValueError("Fenêtre trop courte.")

    hop = max(1, int(round(fen * 0.5)))
    temps_centre = []
    frequence_pic = []
    decibel_SNR = []
    prominence = []

    i = 0
    while i + fen <= N:
        segment = x[i:i + fen]
        freqs, Pxx = puissance_spectre(segment, fs, nfft=4 * fen, window="hann")
        f_pic, SNR_db, prom = qualite_pics(freqs, Pxx, freq_min, freq_max)
        temps_centre.append((i + fen / 2) / fs)
        frequence_pic.append(f_pic)
        decibel_SNR.append(SNR_db)
        prominence.append(prom)
        i += hop

    return (
        np.array(temps_centre),
        np.array(frequence_pic),
        np.array(decibel_SNR),
        np.array(prominence),
    )


def tracking_freq(freq_est, snr_db, prom_norm, saut_max=0.15, alph=0.3, snr_min=3.0, prom_min=0.02):
    f_est = np.asarray(freq_est, dtype=np.float64)
    snr_db = np.asarray(snr_db, dtype=np.float64)
    prom_norm = np.asarray(prom_norm, dtype=np.float64)

    f_track = np.full_like(f_est, np.nan)
    f_prev = np.nan

    for k in range(f_est.size):
        f = f_est[k]
        ok = np.isfinite(f) and snr_db[k] >= snr_min and prom_norm[k] >= prom_min
        if not ok:
            continue

        if np.isfinite(f_prev):
            if abs(f - f_prev) > saut_max:
                # on amortit le saut au lieu de jeter la mesure
                f = 0.75 * f_prev + 0.25 * f
            f_lisse = (1 - alph) * f_prev + alph * f
        else:
            f_lisse = f

        f_track[k] = f_lisse
        f_prev = f_lisse

    return f_track


# =========================
# Estimation RR / HR
# =========================
def estimation_rr(signal_rr, fs, duree_fenetre=20.0):
    t, f, snr, prom = FFT_glissante(signal_rr, fs, RR_MIN_HZ, RR_MAX_HZ, duree_fenetre=duree_fenetre)
    f_tr = tracking_freq(f, snr, prom, saut_max=0.05, alph=0.20, snr_min=3.0, prom_min=0.02)
    return t, f_tr * 60.0, snr, prom, f_tr


def estimation_hr(signal_hr, fs, hr_prev_hz=None, duree_fenetre=18.0):
    if hr_prev_hz is not None and np.isfinite(hr_prev_hz):
        fmin = max(HR_MIN_HZ, hr_prev_hz - HR_HALF_BAND_HZ)
        fmax = min(HR_MAX_HZ, hr_prev_hz + HR_HALF_BAND_HZ)
    else:
        fmin = HR_MIN_HZ
        fmax = HR_MAX_HZ

    if (fmax - fmin) < 0.35:
        centre = hr_prev_hz if (hr_prev_hz is not None and np.isfinite(hr_prev_hz)) else 1.5
        fmin = max(HR_MIN_HZ, centre - 0.20)
        fmax = min(HR_MAX_HZ, centre + 0.20)

    t, f, snr, prom = FFT_glissante(signal_hr, fs, fmin, fmax, duree_fenetre=duree_fenetre)
    f_tr = tracking_freq(
        f, snr, prom,
        saut_max=HR_HARD_JUMP_HZ,
        alph=HR_ALPHA_TRACK,
        snr_min=HR_SNR_MIN,
        prom_min=HR_PROM_MIN,
    )

    if np.all(~np.isfinite(f_tr)):
        t, f, snr, prom = FFT_glissante(signal_hr, fs, HR_MIN_HZ, HR_MAX_HZ, duree_fenetre=duree_fenetre)
        f_tr = tracking_freq(f, snr, prom, saut_max=0.18, alph=0.22, snr_min=2.0, prom_min=0.01)

    return t, f_tr * 60.0, snr, prom, f_tr


def filtre_mediane_simple(valeurs):
    vals = [v for v in valeurs if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.median(vals))


# =========================
# Choix du bin (minimal, conservateur)
# =========================
def init_score_bins(nbins):
    global score_bins
    if score_bins is None or len(score_bins) != nbins:
        score_bins = np.zeros(nbins, dtype=np.float64)


def choisir_idx_stable(z_bin, idx_precedent):
    """
    Correction minimale : on garde l'esprit du code de base, mais avec préférence locale,
    hystérésis et bascule globale seulement si le gain est nettement meilleur.
    """
    global score_bins

    amp = np.abs(z_bin)
    init_score_bins(len(amp))
    score_bins = (1.0 - TARGET_SMOOTH_ALPHA) * score_bins + TARGET_SMOOTH_ALPHA * amp

    if idx_precedent is None:
        return int(np.argmax(score_bins))

    n = len(score_bins)
    left = max(0, idx_precedent - TARGET_LOCAL_RADIUS)
    right = min(n, idx_precedent + TARGET_LOCAL_RADIUS + 1)

    idx_local = left + int(np.argmax(score_bins[left:right]))
    idx_global = int(np.argmax(score_bins))

    prev_gain = score_bins[idx_precedent]
    local_gain = score_bins[idx_local]
    global_gain = score_bins[idx_global]

    if abs(idx_local - idx_precedent) <= TARGET_HYST_BINS and local_gain >= 0.98 * prev_gain:
        return idx_local

    if global_gain >= TARGET_SWITCH_RATIO * prev_gain:
        return idx_global

    return int(idx_precedent)


# =========================
# Motion gating (quasi-statique)
# =========================
def motion_gate_score(phi_hp, fs):
    """
    Inspiré de la piste 'motion gating' du PDF : si la fenêtre récente devient instable,
    on ralentit fortement la mise à jour de la sortie, sans cesser d’émettre une HR.
    """
    if phi_hp.size < max(8, int(MOTION_GATE_SECONDS * fs)):
        return 0.0, True

    n = int(MOTION_GATE_SECONDS * fs)
    x = np.asarray(phi_hp[-n:], dtype=np.float64)
    x = x - np.median(x)
    if x.size < 8:
        return 0.0, True

    var_norm = np.std(x)
    dx = np.diff(x)
    mad_dx = np.median(np.abs(dx - np.median(dx))) / 0.6745 if dx.size else 0.0

    score = max(var_norm / MOTION_VAR_THR, mad_dx / MOTION_DIFF_MAD_THR)
    motion_ok = bool(score < 1.0)
    return float(score), motion_ok


# =========================
# Lissage de sortie
# =========================
def adaptive_alpha(snr_db, prom_norm, hr_target_bpm, hr_prev_bpm, motion_ok):
    if not np.isfinite(snr_db):
        snr_db = 0.0
    if not np.isfinite(prom_norm):
        prom_norm = 0.0

    snr_term = np.clip((snr_db - 2.0) / 8.0, 0.0, 1.0)
    prom_term = np.clip(prom_norm / 0.20, 0.0, 1.0)
    alpha = HR_EMA_ALPHA_MIN + (HR_EMA_ALPHA_MAX - HR_EMA_ALPHA_MIN) * (0.65 * snr_term + 0.35 * prom_term)

    if np.isfinite(hr_prev_bpm) and np.isfinite(hr_target_bpm):
        gap = abs(hr_target_bpm - hr_prev_bpm)
        if gap > 6.0:
            alpha = min(HR_EMA_ALPHA_MAX, alpha + 0.05)

    if not motion_ok:
        alpha = min(alpha, MOTION_FREEZE_ALPHA)

    return float(np.clip(alpha, HR_EMA_ALPHA_MIN, HR_EMA_ALPHA_MAX))


def slew_rate_limit(new_value, old_value, max_step):
    if not np.isfinite(new_value):
        return old_value
    if not np.isfinite(old_value):
        return new_value
    delta = np.clip(new_value - old_value, -max_step, max_step)
    return float(old_value + delta)


# =========================
# Affichage HR
# =========================
def init_hr_plot():
    if not SHOW_HR_GRAPH:
        return None
    plt.ion()
    fig, ax = plt.subplots()
    line_hr, = ax.plot([], [], label="HR lissée")
    line_raw, = ax.plot([], [], label="HR brute", alpha=0.6)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("HR (bpm)")
    ax.set_title("Fréquence cardiaque estimée")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(40, 140)
    return fig, ax, line_hr, line_raw


def update_hr_plot(plot_handles, t_now, hr_bpm, hr_raw_bpm):
    if plot_handles is None:
        return
    fig, ax, line_hr, line_raw = plot_handles
    plot_times.append(t_now)
    plot_hr.append(hr_bpm if np.isfinite(hr_bpm) else np.nan)
    plot_hr_raw.append(hr_raw_bpm if np.isfinite(hr_raw_bpm) else np.nan)

    t0 = plot_times[0]
    xs = np.array(plot_times) - t0
    line_hr.set_data(xs, np.array(plot_hr))
    line_raw.set_data(xs, np.array(plot_hr_raw))

    ax.set_xlim(max(0.0, xs[-1] - 120.0), max(120.0, xs[-1]))
    ys = np.array([v for v in list(plot_hr) + list(plot_hr_raw) if np.isfinite(v)], dtype=float)
    if ys.size:
        ymin = max(40.0, np.min(ys) - 8.0)
        ymax = min(150.0, np.max(ys) + 8.0)
        if ymax - ymin >= 10:
            ax.set_ylim(ymin, ymax)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


# =========================
# Main
# =========================
def main():
    global fs_est, idx_lock, reacq_count, hr_last_output, hr_last_stable_hz, rr_last_output, marqueur_print

    init_radar()
    init_logger()
    plot_handles = init_hr_plot()

    try:
        while True:
            code_erreur, resultat, tableau_IQ = uRAD_RP_SDK11.detection()
            if code_erreur != 0:
                closeProgram()
                break

            I_brut = np.asarray(tableau_IQ[0], dtype=np.float64)
            Q_brut = np.asarray(tableau_IQ[1], dtype=np.float64)
            z_bin = I_brut + 1j * Q_brut

            if idx_lock is None or reacq_count >= TARGET_REACQ_PERIOD:
                init_score_bins(len(z_bin))
                score_bins[:] = np.abs(z_bin).astype(np.float64)
                idx_lock = int(np.argmax(score_bins))
                reacq_count = 0
            else:
                idx_lock = choisir_idx_stable(z_bin, idx_lock)

            reacq_count += 1

            # bin unique pour ne pas écraser la phase utile
            z_sel = z_bin[idx_lock]
            I_sel = float(np.real(z_sel))
            Q_sel = float(np.imag(z_sel))

            # phase CW : prétraitement I/Q, atan2, unwrap
            I_phase, Q_phase = pretraitement_phase(I_sel, Q_sel)
            phase = np.arctan2(Q_phase, I_phase)
            phase_deplie_val = unwrap_phase(phase)

            t_now = time.time()
            buffer_temps.append(t_now)
            buffer_phi.append(phase_deplie_val)

            if len(buffer_temps) >= 2:
                dt_last = buffer_temps[-1] - buffer_temps[-2]
                fs_inst = 1.0 / max(dt_last, 1e-6)
                fs_est = fs_inst if fs_est is None else (1 - beta_fs) * fs_est + beta_fs * fs_inst

            while (buffer_temps[-1] - buffer_temps[0]) > WINDOW_SECONDS:
                buffer_temps.popleft()
                buffer_phi.popleft()

            if len(buffer_phi) < MIN_SAMPLES:
                continue

            tab_phi = np.array(buffer_phi, dtype=np.float64)
            if fs_est is not None:
                fs = fs_est
            else:
                tab_t = np.array(buffer_temps, dtype=np.float64)
                fs = 1.0 / np.median(np.diff(tab_t))

            # suppression dérive / artefacts lents
            phi_hp = filtre_pass_haut(tab_phi, fs, fc=0.05)
            motion_score, motion_ok = motion_gate_score(phi_hp, fs)

            # RR
            rr_out = rr_last_output if np.isfinite(rr_last_output) else np.nan
            rr_f_out = np.nan
            rr_snr_out = np.nan
            rr_prom_out = np.nan
            try:
                sig_rr = passe_bande(phi_hp, RR_MIN_HZ, RR_MAX_HZ, fs)
                t_rr, rr_rpm, rr_snr, rr_prom, rr_f = estimation_rr(sig_rr, fs, duree_fenetre=20.0)
                rr_val = rr_rpm[np.isfinite(rr_rpm)]
                rr_f_val = rr_f[np.isfinite(rr_f)]
                rr_snr_val = rr_snr[np.isfinite(rr_snr)]
                rr_prom_val = rr_prom[np.isfinite(rr_prom)]
                if rr_val.size:
                    rr_out = rr_val[-1]
                    rr_last_output = rr_out
                if rr_f_val.size:
                    rr_f_out = rr_f_val[-1]
                if rr_snr_val.size:
                    rr_snr_out = rr_snr_val[-1]
                if rr_prom_val.size:
                    rr_prom_out = rr_prom_val[-1]
            except Exception:
                pass

            # HR : dérivée de phase + bande cardiaque + rejet harmoniques RR
            phi_diff = np.diff(phi_hp, prepend=phi_hp[0])
            try:
                sig_hr = passe_bande(phi_diff, HR_MIN_HZ, HR_MAX_HZ, fs)
                sig_hr = notch_resp_harmonics(sig_hr, fs, rr_f_out, max_harm=6, bw=0.06)
            except Exception:
                sig_hr = phi_diff.copy()

            if (t_now - marqueur_print) > PRINT_PERIOD:
                hr_prev = hr_last_stable_hz if np.isfinite(hr_last_stable_hz) else None
                hr_out_raw = np.nan
                hr_f_out = np.nan
                hr_snr_out = np.nan
                hr_prom_out = np.nan
                hr_med = np.nan
                alpha = HR_EMA_ALPHA_MIN

                try:
                    t_hr, hr_bpm, hr_snr, hr_prom, hr_f = estimation_hr(
                        sig_hr, fs, hr_prev_hz=hr_prev, duree_fenetre=18.0
                    )
                    hr_val = hr_bpm[np.isfinite(hr_bpm)]
                    hr_f_val = hr_f[np.isfinite(hr_f)]
                    hr_snr_val = hr_snr[np.isfinite(hr_snr)]
                    hr_prom_val = hr_prom[np.isfinite(hr_prom)]

                    if hr_val.size:
                        hr_out_raw = hr_val[-1]
                        hr_history.append(hr_out_raw)
                        hr_med = filtre_mediane_simple(hr_history)
                    if hr_f_val.size:
                        hr_f_out = hr_f_val[-1]
                    if hr_snr_val.size:
                        hr_snr_out = hr_snr_val[-1]
                    if hr_prom_val.size:
                        hr_prom_out = hr_prom_val[-1]
                except Exception:
                    pass

                if np.isfinite(hr_med):
                    alpha = adaptive_alpha(hr_snr_out, hr_prom_out, hr_med, hr_last_output, motion_ok)
                    if np.isfinite(hr_last_output):
                        hr_target = (1.0 - alpha) * hr_last_output + alpha * hr_med
                    else:
                        hr_target = hr_med
                    hr_last_output = slew_rate_limit(hr_target, hr_last_output, HR_MAX_STEP_BPM)
                    hr_last_stable_hz = hr_last_output / 60.0

                hr_print = hr_last_output if np.isfinite(hr_last_output) else np.nan

                print(
                    f"RR: {rr_out:.2f} rpm | "
                    f"HR: {hr_print:.2f} bpm | "
                    f"HR_brut: {hr_out_raw:.2f} bpm | "
                    f"SNR_HR: {hr_snr_out:.2f} dB | "
                    f"PROM_HR: {hr_prom_out:.4f} | "
                    f"motion: {motion_score:.2f} ({'OK' if motion_ok else 'FREEZE'}) | "
                    f"HR_ref_hz: {hr_last_stable_hz:.3f} | "
                    f"bin: {idx_lock}"
                )

                write_log([
                    t_now, fs, idx_lock, motion_score, int(motion_ok),
                    rr_out, rr_f_out, rr_snr_out, rr_prom_out,
                    hr_print, hr_out_raw, hr_med, hr_last_stable_hz,
                    hr_snr_out, hr_prom_out, alpha, HR_MAX_STEP_BPM,
                ])
                update_hr_plot(plot_handles, t_now, hr_print, hr_out_raw)
                marqueur_print = t_now

    except KeyboardInterrupt:
        closeProgram()


if __name__ == "__main__":
    main()
