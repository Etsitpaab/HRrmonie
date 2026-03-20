import time
import csv
import os
from collections import deque

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, get_window
import matplotlib.pyplot as plt
import uRAD_RP_SDK11

# =========================
# Configuration radar uRAD
# =========================
mode = 1  # CW
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
# Paramètres HR / RR
# =========================
HR_MIN_HZ = 0.90
HR_MAX_HZ = 2.00
HR_HALF_BAND_HZ = 0.35
HR_HARD_JUMP_HZ = 0.18
HR_ALPHA_TRACK = 0.12
HR_SNR_MIN = 2.5
HR_PROM_MIN = 0.015
HR_MEDIAN_LEN = 5
HR_MAX_INVALID_STREAK = 6
HR_MAX_STEP_BPM = 2.5          # variation max par mise à jour affichée
HR_ALPHA_MIN = 0.08
HR_ALPHA_MAX = 0.30
HR_ALPHA_BASE = 0.12
HR_MAX_HARM_REJECT = 8

RR_MIN_HZ = 0.10
RR_MAX_HZ = 0.50

TARGET_SMOOTH_ALPHA = 0.15
TARGET_REACQ_PERIOD = 120
TARGET_NEIGHBOR_MARGIN = 1
TARGET_LOCAL_MARGIN = 2
TARGET_SWITCH_RATIO = 1.10
TARGET_GLOBAL_CONFIRM = 3
TARGET_FAR_JUMP_MAX = 4

# =========================
# Logging
# =========================
LOG_ENABLED = True
LOG_DIR = "logs_hr"
LOG_PREFIX = "vitals_log"
LOG_FLUSH_EVERY = 1

log_writer = None
log_file_handle = None
log_path = None
log_counter = 0


def init_logging():
    global log_writer, log_file_handle, log_path
    if not LOG_ENABLED:
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{LOG_PREFIX}_{ts}.csv")
    log_file_handle = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file_handle)
    log_writer.writerow([
        "timestamp",
        "fs",
        "idx_lock",
        "idx_local_best",
        "idx_global_best",
        "rr_rpm",
        "rr_hz",
        "rr_snr_db",
        "hr_out_bpm",
        "hr_raw_bpm",
        "hr_med_bpm",
        "hr_peak_freq_hz",
        "hr_peak_snr_db",
        "hr_peak_prom",
        "hr_alpha",
        "hr_delta_before_limit",
        "hr_step_applied",
        "phase_last",
        "bin_amp",
        "reacq_count",
        "invalid_hr_streak",
    ])
    log_file_handle.flush()
    print(f"[LOG] CSV: {log_path}")


def write_log(row):
    global log_counter
    if not LOG_ENABLED or log_writer is None:
        return
    log_writer.writerow(row)
    log_counter += 1
    if log_counter % LOG_FLUSH_EVERY == 0:
        log_file_handle.flush()


# =========================
# Utilitaires radar
# =========================
def closeProgram():
    global log_file_handle
    try:
        if log_file_handle is not None:
            log_file_handle.flush()
            log_file_handle.close()
            log_file_handle = None
    finally:
        uRAD_RP_SDK11.turnOFF()
    raise SystemExit


return_code = uRAD_RP_SDK11.turnON()
if return_code != 0:
    closeProgram()

return_code = uRAD_RP_SDK11.loadConfiguration(
    mode, f0, BW, Ns, Ntar, Rmax, MTI, Mth, Alpha,
    distance_true, velocity_true, SNR_true,
    I_true, Q_true, movement_true
)
if return_code != 0:
    closeProgram()

init_logging()

# =========================
# Prétraitement pour la phase
# =========================
beta_dc_phase = 0.01
moyenne_I_phase = None
moyenne_Q_phase = None


def pretraitement_phase(I_entree, Q_entree):
    global moyenne_I_phase, moyenne_Q_phase

    I_entree = float(I_entree)
    Q_entree = float(Q_entree)

    if moyenne_I_phase is None:
        moyenne_I_phase = I_entree
        moyenne_Q_phase = Q_entree

    moyenne_I_phase = (1 - beta_dc_phase) * moyenne_I_phase + beta_dc_phase * I_entree
    moyenne_Q_phase = (1 - beta_dc_phase) * moyenne_Q_phase + beta_dc_phase * Q_entree

    I_centre = I_entree - moyenne_I_phase
    Q_centre = Q_entree - moyenne_Q_phase
    return I_centre, Q_centre


# =========================
# Dépliage de phase
# =========================
phase_deplie = 0.0
phase_precedente = None


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
def filtre_pass_haut(signal, freq_echanti, fc=0.05):
    signal = np.asarray(signal, dtype=np.float64)
    nyq = freq_echanti / 2.0
    if fc >= nyq:
        return signal.copy()
    sos = butter(4, fc / nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, signal)



def passe_bande(signal, f_basse, f_haut, freq_echanti, ordre=4):
    signal = np.asarray(signal, dtype=np.float64)
    nyq = freq_echanti / 2.0
    if f_basse <= 0 or f_haut >= nyq or f_basse >= f_haut:
        raise ValueError("Bornes de passe-bande invalides.")
    sos = butter(ordre, [f_basse / nyq, f_haut / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)



def notch_resp_harmonics(signal, fs, f_rr, max_harm=HR_MAX_HARM_REJECT, bw=0.05):
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
# Spectre et tracking
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



def FFT_glissante(x, freq, freq_min, freq_max, duree_fenetre=20.0):
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    fen = int(round(duree_fenetre * freq))
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
        freqs, Pxx = puissance_spectre(segment, freq, nfft=4 * fen, window="hann")
        f_pic, SNR_db, prom = qualite_pics(freqs, Pxx, freq_min, freq_max)
        temps_centre.append((i + fen / 2) / freq)
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
            delta = abs(f - f_prev)
            if delta > saut_max:
                f = 0.7 * f_prev + 0.3 * f
            f_lisse = (1.0 - alph) * f_prev + alph * f
        else:
            f_lisse = f

        f_track[k] = f_lisse
        f_prev = f_lisse

    return f_track


# =========================
# Estimation RR / HR
# =========================
def estimation_rr(signal_rr, fs, duree_fenetre=20.0):
    t, f, snr, prom = FFT_glissante(
        signal_rr, fs,
        freq_min=RR_MIN_HZ,
        freq_max=RR_MAX_HZ,
        duree_fenetre=duree_fenetre,
    )
    f_tr = tracking_freq(f, snr, prom, saut_max=0.05, alph=0.20, snr_min=3.0, prom_min=0.02)
    rpm = f_tr * 60.0
    return t, rpm, snr, prom, f_tr



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

    t, f, snr, prom = FFT_glissante(
        signal_hr, fs,
        freq_min=fmin,
        freq_max=fmax,
        duree_fenetre=duree_fenetre,
    )
    f_tr = tracking_freq(
        f, snr, prom,
        saut_max=HR_HARD_JUMP_HZ,
        alph=HR_ALPHA_TRACK,
        snr_min=HR_SNR_MIN,
        prom_min=HR_PROM_MIN,
    )

    if np.all(~np.isfinite(f_tr)):
        t, f, snr, prom = FFT_glissante(
            signal_hr, fs,
            freq_min=HR_MIN_HZ,
            freq_max=HR_MAX_HZ,
            duree_fenetre=duree_fenetre,
        )
        f_tr = tracking_freq(
            f, snr, prom,
            saut_max=0.18,
            alph=0.20,
            snr_min=2.0,
            prom_min=0.01,
        )

    bpm = f_tr * 60.0
    return t, bpm, snr, prom, f_tr



def filtre_mediane_simple(valeurs):
    vals = [v for v in valeurs if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.median(vals))



def adaptive_alpha(snr_db, prom_norm, delta_bpm):
    alpha = HR_ALPHA_BASE

    if np.isfinite(snr_db):
        alpha += 0.012 * np.clip(snr_db - 3.0, 0.0, 8.0)

    if np.isfinite(prom_norm):
        alpha += 0.10 * np.clip(prom_norm, 0.0, 1.0)

    if np.isfinite(delta_bpm):
        if delta_bpm > 8.0:
            alpha += 0.05
        elif delta_bpm < 2.0:
            alpha -= 0.02

    return float(np.clip(alpha, HR_ALPHA_MIN, HR_ALPHA_MAX))



def slew_rate_limit(prev_bpm, target_bpm, max_step_bpm=HR_MAX_STEP_BPM):
    if not np.isfinite(target_bpm):
        return prev_bpm, 0.0, np.nan
    if not np.isfinite(prev_bpm):
        return target_bpm, 0.0, 0.0

    delta = target_bpm - prev_bpm
    step = float(np.clip(delta, -max_step_bpm, max_step_bpm))
    return prev_bpm + step, step, delta


# =========================
# Graphe HR mesurée
# =========================
plt.ion()
fig, ax = plt.subplots()
line_hr_out, = ax.plot([], [], label="HR")
line_hr_raw, = ax.plot([], [], label="HR brut")
ax.set_xlabel("Temps (s)")
ax.set_ylabel("HR (bpm)")
ax.set_title("HR mesurée")
ax.grid(True)
ax.legend(loc="upper right")

hr_plot_t = deque(maxlen=600)
hr_plot_out = deque(maxlen=600)
hr_plot_raw = deque(maxlen=600)
plot_t0 = time.time()


# =========================
# Buffers et états
# =========================
seconde_fenetre = 30.0
echantillon_min = 200
affichage = 1.0

buffer_temps = deque()
buffer_phi = deque()

marqueur_print = time.time()

fs_est = None
beta_fs = 0.05

idx_lock = None
reacq_count = 0
score_bins = None
global_switch_streak = 0
last_global_candidate = None
last_local_best = None
last_global_best = None

hr_history = deque(maxlen=HR_MEDIAN_LEN)
hr_last_stable_hz = np.nan
hr_last_output = np.nan
invalid_hr_streak = 0


def init_score_bins(nbins):
    global score_bins
    if score_bins is None or len(score_bins) != nbins:
        score_bins = np.zeros(nbins, dtype=np.float64)



def choisir_idx_stable(z_bin, idx_precedent):
    global score_bins, global_switch_streak, last_global_candidate, last_local_best, last_global_best

    amp = np.abs(z_bin)
    init_score_bins(len(amp))
    score_bins = (1.0 - TARGET_SMOOTH_ALPHA) * score_bins + TARGET_SMOOTH_ALPHA * amp

    if idx_precedent is None:
        best = int(np.argmax(score_bins))
        last_local_best = best
        last_global_best = best
        return best

    n = len(score_bins)
    left = max(0, idx_precedent - TARGET_LOCAL_MARGIN)
    right = min(n, idx_precedent + TARGET_LOCAL_MARGIN + 1)

    idx_local = left + int(np.argmax(score_bins[left:right]))
    idx_global = int(np.argmax(score_bins))
    last_local_best = idx_local
    last_global_best = idx_global

    prev_score = score_bins[idx_precedent] + 1e-12
    local_score = score_bins[idx_local]
    global_score = score_bins[idx_global]

    if local_score >= 0.92 * prev_score:
        global_switch_streak = 0
        last_global_candidate = None
        return idx_local

    far_jump = abs(idx_global - idx_precedent) > TARGET_FAR_JUMP_MAX
    strong_global = global_score >= TARGET_SWITCH_RATIO * max(local_score, prev_score)

    if far_jump and strong_global:
        if last_global_candidate == idx_global:
            global_switch_streak += 1
        else:
            last_global_candidate = idx_global
            global_switch_streak = 1

        if global_switch_streak >= TARGET_GLOBAL_CONFIRM:
            global_switch_streak = 0
            last_global_candidate = None
            return idx_global
        return idx_precedent

    global_switch_streak = 0
    last_global_candidate = None

    if local_score >= prev_score:
        return idx_local
    return idx_precedent


# =========================
# Boucle principale
# =========================
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
            score_bins = np.abs(z_bin).astype(np.float64)
            idx_lock = int(np.argmax(score_bins))
            reacq_count = 0
            global_switch_streak = 0
        else:
            idx_lock = choisir_idx_stable(z_bin, idx_lock)

        reacq_count += 1

        left = max(0, idx_lock - TARGET_NEIGHBOR_MARGIN)
        right = min(len(z_bin), idx_lock + TARGET_NEIGHBOR_MARGIN + 1)
        z_roi = z_bin[left:right]
        roi_amp = np.abs(z_roi)
        roi_w = roi_amp / (np.sum(roi_amp) + 1e-12)
        z_sel = np.sum(z_roi * roi_w)

        I_sel = float(np.real(z_sel))
        Q_sel = float(np.imag(z_sel))

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

        while (buffer_temps[-1] - buffer_temps[0]) > seconde_fenetre:
            buffer_temps.popleft()
            buffer_phi.popleft()

        if len(buffer_phi) < echantillon_min:
            continue

        tab_phi = np.array(buffer_phi, dtype=np.float64)
        if fs_est is not None:
            fs = fs_est
        else:
            tab_t = np.array(buffer_temps, dtype=np.float64)
            dt = np.diff(tab_t)
            fs = 1.0 / np.median(dt)

        phi_hp = filtre_pass_haut(tab_phi, fs, fc=0.05)

        rr_snr_out = np.nan
        try:
            sig_rr = passe_bande(phi_hp, RR_MIN_HZ, RR_MAX_HZ, fs)
            t_rr, rr_rpm, rr_snr, rr_prom, rr_f = estimation_rr(sig_rr, fs, duree_fenetre=20.0)
            rr_val = rr_rpm[np.isfinite(rr_rpm)]
            rr_out = rr_val[-1] if rr_val.size else np.nan

            rr_f_val = rr_f[np.isfinite(rr_f)]
            rr_f_out = rr_f_val[-1] if rr_f_val.size else np.nan

            rr_snr_val = rr_snr[np.isfinite(rr_snr)]
            rr_snr_out = rr_snr_val[-1] if rr_snr_val.size else np.nan
        except Exception:
            rr_out = np.nan
            rr_f_out = np.nan

        phi_diff = np.diff(phi_hp, prepend=phi_hp[0])

        try:
            sig_hr = passe_bande(phi_diff, HR_MIN_HZ, HR_MAX_HZ, fs)
            sig_hr = notch_resp_harmonics(sig_hr, fs, rr_f_out, max_harm=HR_MAX_HARM_REJECT, bw=0.05)
        except Exception:
            sig_hr = phi_diff.copy()

        if (t_now - marqueur_print) <= affichage:
            continue

        hr_prev = hr_last_stable_hz if np.isfinite(hr_last_stable_hz) else None
        hr_med = np.nan
        hr_alpha = 0.0
        hr_delta_before_limit = np.nan
        hr_step_applied = 0.0

        try:
            t_hr, hr_bpm, hr_snr, hr_prom, hr_f = estimation_hr(
                sig_hr,
                fs,
                hr_prev_hz=hr_prev,
                duree_fenetre=18.0,
            )

            hr_val = hr_bpm[np.isfinite(hr_bpm)]
            hr_f_val = hr_f[np.isfinite(hr_f)]
            hr_snr_val = hr_snr[np.isfinite(hr_snr)]
            hr_prom_val = hr_prom[np.isfinite(hr_prom)]

            hr_out_raw = hr_val[-1] if hr_val.size else np.nan
            hr_f_out = hr_f_val[-1] if hr_f_val.size else np.nan
            hr_snr_out = hr_snr_val[-1] if hr_snr_val.size else np.nan
            hr_prom_out = hr_prom_val[-1] if hr_prom_val.size else np.nan
        except Exception:
            hr_out_raw = np.nan
            hr_f_out = np.nan
            hr_snr_out = np.nan
            hr_prom_out = np.nan

        if np.isfinite(hr_out_raw):
            invalid_hr_streak = 0
            hr_history.append(hr_out_raw)
            hr_med = filtre_mediane_simple(hr_history)

            if np.isfinite(hr_med):
                delta_bpm = abs(hr_med - hr_last_output) if np.isfinite(hr_last_output) else 0.0
                hr_alpha = adaptive_alpha(hr_snr_out, hr_prom_out, delta_bpm)

                if np.isfinite(hr_last_output):
                    hr_target = (1.0 - hr_alpha) * hr_last_output + hr_alpha * hr_med
                else:
                    hr_target = hr_med

                hr_last_output, hr_step_applied, hr_delta_before_limit = slew_rate_limit(
                    hr_last_output,
                    hr_target,
                    max_step_bpm=HR_MAX_STEP_BPM,
                )
                hr_last_stable_hz = hr_last_output / 60.0
        else:
            invalid_hr_streak += 1
            if invalid_hr_streak >= HR_MAX_INVALID_STREAK:
                hr_history.clear()

        hr_print = hr_last_output if np.isfinite(hr_last_output) else np.nan

        # Mise à jour graphe HR
        t_plot = t_now - plot_t0
        hr_plot_t.append(t_plot)
        hr_plot_out.append(hr_print)
        hr_plot_raw.append(hr_out_raw)

        line_hr_out.set_data(hr_plot_t, hr_plot_out)
        line_hr_raw.set_data(hr_plot_t, hr_plot_raw)
        if hr_plot_t:
            ax.set_xlim(max(0.0, hr_plot_t[0]), hr_plot_t[-1] + 1.0)
            valid_vals = [v for v in list(hr_plot_out) + list(hr_plot_raw) if np.isfinite(v)]
            if valid_vals:
                vmin = max(40.0, min(valid_vals) - 5.0)
                vmax = min(140.0, max(valid_vals) + 5.0)
                if vmax <= vmin:
                    vmax = vmin + 10.0
                ax.set_ylim(vmin, vmax)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

        print(
            f"RR: {rr_out:.2f} rpm | "
            f"HR: {hr_print:.2f} bpm | "
            f"HR_brut: {hr_out_raw:.2f} bpm | "
            f"HR_med: {hr_med:.2f} bpm | "
            f"SNR_HR: {hr_snr_out:.2f} dB | "
            f"PROM_HR: {hr_prom_out:.4f} | "
            f"ALPHA_HR: {hr_alpha:.2f} | "
            f"STEP_HR: {hr_step_applied:.2f} | "
            f"HR_ref_hz: {hr_last_stable_hz:.3f} | "
            f"bin: {idx_lock}"
        )

        write_log([
            t_now,
            fs,
            idx_lock,
            last_local_best,
            last_global_best,
            rr_out,
            rr_f_out,
            rr_snr_out,
            hr_print,
            hr_out_raw,
            hr_med,
            hr_f_out,
            hr_snr_out,
            hr_prom_out,
            hr_alpha,
            hr_delta_before_limit,
            hr_step_applied,
            phase_deplie_val,
            float(np.abs(z_bin[idx_lock])) if 0 <= idx_lock < len(z_bin) else np.nan,
            reacq_count,
            invalid_hr_streak,
        ])

        marqueur_print = t_now

except KeyboardInterrupt:
    closeProgram()
