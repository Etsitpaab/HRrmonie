import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, get_window
import uRAD_RP_SDK11
import matplotlib.pyplot as plt
from collections import deque

# =========================
# Configuration radar uRAD
# =========================
mode = 1
f0 = 125
BW = 240
Ns = 200
Ntar = 3   # 1 à 5
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
HR_MIN_HZ = 0.90          # 54 bpm
HR_MAX_HZ = 2.00          # 120 bpm
HR_HALF_BAND_HZ = 0.35    # bande adaptative +/- 21 bpm
HR_HARD_JUMP_HZ = 0.12    # tracking moins strict
HR_ALPHA_TRACK = 0.12
HR_SNR_MIN = 2.5
HR_PROM_MIN = 0.015
HR_MEDIAN_LEN = 7
HR_EMA_ALPHA = 0.25

RR_MIN_HZ = 0.10
RR_MAX_HZ = 0.50

TARGET_SMOOTH_ALPHA = 0.15
TARGET_REACQ_PERIOD = 120
TARGET_NEIGHBOR_MARGIN = 1

# =========================
# Utilitaires radar
# =========================
def closeProgram():
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

# =========================
# Prétraitement pour affichage IQ
# =========================
beta_dc_iq = 0.001
beta_variance_iq = 0.001
epsilon = 1e-6

moyenne_I_iq = None
moyenne_Q_iq = None
variance_I_iq = None
variance_Q_iq = None

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

    sos = butter(4, fc / nyq, btype='highpass', output='sos')
    return sosfiltfilt(sos, signal)

def passe_bande(signal, f_basse, f_haut, freq_echanti, ordre=4):
    signal = np.asarray(signal, dtype=np.float64)
    nyq = freq_echanti / 2.0

    if f_basse <= 0 or f_haut >= nyq or f_basse >= f_haut:
        raise ValueError("Bornes de passe-bande invalides.")

    sos = butter(ordre, [f_basse / nyq, f_haut / nyq], btype='bandpass', output='sos')
    return sosfiltfilt(sos, signal)

def notch_resp_harmonics(signal, fs, f_rr, max_harm=6, bw=0.06):
    """
    Rejet simple des harmoniques respiratoires.
    """
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
        np.array(prominence)
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
                continue
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
    t, f, snr, prom = FFT_glissante(
        signal_rr, fs,
        freq_min=RR_MIN_HZ,
        freq_max=RR_MAX_HZ,
        duree_fenetre=duree_fenetre
    )
    f_tr = tracking_freq(f, snr, prom, saut_max=0.05, alph=0.20, snr_min=3.0, prom_min=0.02)
    rpm = f_tr * 60.0
    return t, rpm, snr, prom, f_tr

def estimation_hr(signal_hr, fs, hr_prev_hz=None, duree_fenetre=18.0):
    # 1) Bande adaptative
    if hr_prev_hz is not None and np.isfinite(hr_prev_hz):
        fmin = max(HR_MIN_HZ, hr_prev_hz - HR_HALF_BAND_HZ)
        fmax = min(HR_MAX_HZ, hr_prev_hz + HR_HALF_BAND_HZ)
    else:
        fmin = HR_MIN_HZ
        fmax = HR_MAX_HZ

    # Sécurité sur bande trop étroite
    if (fmax - fmin) < 0.35:
        centre = hr_prev_hz if (hr_prev_hz is not None and np.isfinite(hr_prev_hz)) else 1.5
        fmin = max(HR_MIN_HZ, centre - 0.20)
        fmax = min(HR_MAX_HZ, centre + 0.20)

    # 2) Essai en bande adaptative
    t, f, snr, prom = FFT_glissante(
        signal_hr, fs,
        freq_min=fmin,
        freq_max=fmax,
        duree_fenetre=duree_fenetre
    )
    f_tr = tracking_freq(
        f, snr, prom,
        saut_max=HR_HARD_JUMP_HZ,
        alph=HR_ALPHA_TRACK,
        snr_min=HR_SNR_MIN,
        prom_min=HR_PROM_MIN
    )

    # 3) Fallback large bande si tout est NaN
    if np.all(~np.isfinite(f_tr)):
        t, f, snr, prom = FFT_glissante(
            signal_hr, fs,
            freq_min=HR_MIN_HZ,
            freq_max=HR_MAX_HZ,
            duree_fenetre=duree_fenetre
        )
        f_tr = tracking_freq(
            f, snr, prom,
            saut_max=0.18,
            alph=0.20,
            snr_min=2.0,
            prom_min=0.01
        )

    bpm = f_tr * 60.0
    return t, bpm, snr, prom, f_tr

def filtre_mediane_simple(valeurs):
    vals = [v for v in valeurs if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.median(vals))

# =========================
# Affichage constellation
# =========================
plt.ion()

fig, ax = plt.subplots()
sc = ax.scatter([], [], s=6)

ax.set_xlabel("I")
ax.set_ylabel("Q")
ax.set_title("Constellation I/Q")
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

histo_i = deque(maxlen=4000)
histo_q = deque(maxlen=4000)

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

hr_history = deque(maxlen=HR_MEDIAN_LEN)
hr_last_stable_hz = np.nan
hr_last_output = np.nan

def init_score_bins(nbins):
    global score_bins
    if score_bins is None or len(score_bins) != nbins:
        score_bins = np.zeros(nbins, dtype=np.float64)

def choisir_idx_stable(z_bin, idx_precedent):
    global score_bins

    amp = np.abs(z_bin)
    init_score_bins(len(amp))

    score_bins = (1.0 - TARGET_SMOOTH_ALPHA) * score_bins + TARGET_SMOOTH_ALPHA * amp

    if idx_precedent is None:
        return int(np.argmax(score_bins))

    n = len(score_bins)
    left = max(0, idx_precedent - 2)
    right = min(n, idx_precedent + 3)

    idx_local = left + int(np.argmax(score_bins[left:right]))
    local_gain = score_bins[idx_local]
    prev_gain = score_bins[idx_precedent]

    if local_gain >= 1.05 * prev_gain:
        return idx_local

    return int(idx_precedent)

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

        # Verrouillage de cible plus stable
        if idx_lock is None or reacq_count >= TARGET_REACQ_PERIOD:
            init_score_bins(len(z_bin))
            score_bins = np.abs(z_bin).astype(np.float64)
            idx_lock = int(np.argmax(score_bins))
            reacq_count = 0
        else:
            idx_lock = choisir_idx_stable(z_bin, idx_lock)

        reacq_count += 1

        # Moyenne locale autour du bin verrouillé
        left = max(0, idx_lock - TARGET_NEIGHBOR_MARGIN)
        right = min(len(z_bin), idx_lock + TARGET_NEIGHBOR_MARGIN + 1)
        z_roi = z_bin[left:right]
        z_sel = np.mean(z_roi)

        I_sel = float(np.real(z_sel))
        Q_sel = float(np.imag(z_sel))

        # Constellation IQ
        I_aff, Q_aff = pretraitement_affichage_iq(I_sel, Q_sel)
        z_aff = I_aff + 1j * Q_aff

        if abs(z_aff) > 0:
            z_aff = z_aff / abs(z_aff)

        histo_i.append(float(np.real(z_aff)))
        histo_q.append(float(np.imag(z_aff)))

        points = np.c_[histo_i, histo_q]
        sc.set_offsets(points)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

        # Phase physiologique
        I_phase, Q_phase = pretraitement_phase(I_sel, Q_sel)
        phase = np.arctan2(Q_phase, I_phase)
        phase_deplie_val = unwrap_phase(phase)

        t_now = time.time()
        buffer_temps.append(t_now)
        buffer_phi.append(phase_deplie_val)

        # Estimation fs lissée
        if len(buffer_temps) >= 2:
            dt_last = buffer_temps[-1] - buffer_temps[-2]
            fs_inst = 1.0 / max(dt_last, 1e-6)
            fs_est = fs_inst if fs_est is None else (1 - beta_fs) * fs_est + beta_fs * fs_inst

        # Fenêtre glissante
        while (buffer_temps[-1] - buffer_temps[0]) > seconde_fenetre:
            buffer_temps.popleft()
            buffer_phi.popleft()

        # Traitement fréquentiel
        if len(buffer_phi) >= echantillon_min:
            tab_phi = np.array(buffer_phi, dtype=np.float64)

            if fs_est is not None:
                fs = fs_est
            else:
                tab_t = np.array(buffer_temps, dtype=np.float64)
                dt = np.diff(tab_t)
                fs = 1.0 / np.median(dt)

            # Passe-haut lent
            phi_hp = filtre_pass_haut(tab_phi, fs, fc=0.05)

            # RR
            try:
                sig_rr = passe_bande(phi_hp, RR_MIN_HZ, RR_MAX_HZ, fs)
                t_rr, rr_rpm, rr_snr, rr_prom, rr_f = estimation_rr(sig_rr, fs, duree_fenetre=20.0)
                rr_val = rr_rpm[np.isfinite(rr_rpm)]
                rr_out = rr_val[-1] if rr_val.size else np.nan

                rr_f_val = rr_f[np.isfinite(rr_f)]
                rr_f_out = rr_f_val[-1] if rr_f_val.size else np.nan
            except Exception:
                rr_out = np.nan
                rr_f_out = np.nan

            # Dérivée de phase pour accentuer HR
            phi_diff = np.diff(phi_hp, prepend=phi_hp[0])

            try:
                sig_hr = passe_bande(phi_diff, HR_MIN_HZ, HR_MAX_HZ, fs)
                sig_hr = notch_resp_harmonics(sig_hr, fs, rr_f_out, max_harm=5, bw=0.06)
            except Exception:
                sig_hr = phi_diff.copy()

            if (t_now - marqueur_print) > affichage:
                hr_prev = hr_last_stable_hz if np.isfinite(hr_last_stable_hz) else None

                try:
                    t_hr, hr_bpm, hr_snr, hr_prom, hr_f = estimation_hr(
                        sig_hr, fs,
                        hr_prev_hz=hr_prev,
                        duree_fenetre=18.0
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

                # Mise à jour plus permissive pour éviter le gel prolongé
                if np.isfinite(hr_out_raw):
                    hr_history.append(hr_out_raw)
                    hr_med = filtre_mediane_simple(hr_history)

                    if np.isfinite(hr_med):
                        if np.isfinite(hr_last_output):
                            hr_last_output = (1.0 - HR_EMA_ALPHA) * hr_last_output + HR_EMA_ALPHA * hr_med
                        else:
                            hr_last_output = hr_med

                        hr_last_stable_hz = hr_last_output / 60.0

                hr_print = hr_last_output if np.isfinite(hr_last_output) else np.nan

                print(
                    f"RR: {rr_out:.2f} rpm | "
                    f"HR: {hr_print:.2f} bpm | "
                    f"HR_brut: {hr_out_raw:.2f} bpm | "
                    f"SNR_HR: {hr_snr_out:.2f} dB | "
                    f"PROM_HR: {hr_prom_out:.4f} | "
                    f"HR_ref_hz: {hr_last_stable_hz:.3f} | "
                    f"bin: {idx_lock}"
                )

                marqueur_print = t_now

except KeyboardInterrupt:
    closeProgram()