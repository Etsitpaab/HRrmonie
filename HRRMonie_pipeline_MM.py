import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, get_window, find_peaks
import uRAD_RP_SDK11
from collections import deque


# ============================================================
# Configuration radar uRAD
# ============================================================
mode = 1
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


def close_program():
    uRAD_RP_SDK11.turnOFF()
    raise SystemExit


return_code = uRAD_RP_SDK11.turnON()
if return_code != 0:
    close_program()

return_code = uRAD_RP_SDK11.loadConfiguration(
    mode, f0, BW, Ns, Ntar, Rmax, MTI, Mth, Alpha,
    distance_true, velocity_true, SNR_true,
    I_true, Q_true, movement_true
)
if return_code != 0:
    close_program()


# ============================================================
# Prétraitement IQ
# ============================================================
beta_dc = 0.01
mean_I = None
mean_Q = None


def pretraitement_iq(I_in, Q_in):
    global mean_I, mean_Q

    I_in = float(I_in)
    Q_in = float(Q_in)

    if mean_I is None:
        mean_I = I_in
        mean_Q = Q_in

    mean_I = (1 - beta_dc) * mean_I + beta_dc * I_in
    mean_Q = (1 - beta_dc) * mean_Q + beta_dc * Q_in

    return I_in - mean_I, Q_in - mean_Q


# ============================================================
# Filtres
# ============================================================
def filtre_passe_haut(signal, fs, fc=0.05, ordre=4):
    x = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0
    if len(x) < 16 or fc <= 0 or fc >= nyq:
        return x.copy()
    sos = butter(ordre, fc / nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, x)


def filtre_passe_bande(signal, f_low, f_high, fs, ordre=4):
    x = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0
    if len(x) < 16:
        return x.copy()
    if f_low <= 0 or f_high >= nyq or f_low >= f_high:
        return x.copy()
    sos = butter(ordre, [f_low / nyq, f_high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


# ============================================================
# FFT / pics spectraux
# ============================================================
def spectre_fft(x, fs, nfft=None, window="hann"):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    if N < 8:
        return np.array([]), np.array([])

    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(N)))

    w = get_window(window, N)
    xw = (x - np.mean(x)) * w
    X = np.fft.rfft(xw, n=nfft)
    P = np.abs(X) ** 2
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)

    return f, P


def frequence_dominante(signal, fs, fmin, fmax):
    f, P = spectre_fft(signal, fs)
    if f.size == 0:
        return np.nan

    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return np.nan

    f_band = f[mask]
    P_band = P[mask]
    idx = np.argmax(P_band)
    return float(f_band[idx])


def extraire_pics_spectraux(signal, fs, fmin, fmax, max_peaks=6):
    f, P = spectre_fft(signal, fs)
    if f.size == 0:
        return []

    mask = (f >= fmin) & (f <= fmax)
    f_band = f[mask]
    P_band = P[mask]

    if len(P_band) < 3:
        return []

    bruit = np.median(P_band) + 1e-12
    peaks, _ = find_peaks(P_band)

    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(P_band))])

    peaks = peaks[np.argsort(P_band[peaks])[::-1]]

    candidats = []
    for k in peaks[:max_peaks]:
        power = float(P_band[k])
        snr = float(10.0 * np.log10((power + 1e-12) / bruit))
        candidats.append({
            "freq": float(f_band[k]),
            "power": power,
            "snr": snr
        })

    return candidats


# ============================================================
# Range gating renforcé
# ============================================================
idx_lock = None
lock_margin = 1
perte_compteur = 0
perte_max = 20
init_frames = 50
init_bins = []


def initialiser_bin(z_bin):
    global idx_lock, init_bins

    idx = int(np.argmax(np.abs(z_bin)))
    init_bins.append(idx)

    if len(init_bins) >= init_frames:
        hist = np.bincount(init_bins, minlength=len(z_bin))
        idx_lock = int(np.argmax(hist))
        init_bins = []


def selection_bin_verrouillee(z_bin):
    global idx_lock, perte_compteur

    amp = np.abs(z_bin)
    n = len(amp)

    if idx_lock is None:
        initialiser_bin(z_bin)
        idx_tmp = int(np.argmax(amp))
        return z_bin[idx_tmp], idx_tmp

    left = max(0, idx_lock - lock_margin)
    right = min(n, idx_lock + lock_margin + 1)

    idx_local = left + int(np.argmax(amp[left:right]))
    amp_local = amp[idx_local]
    amp_global = np.max(amp)

    if amp_local < 0.50 * amp_global:
        perte_compteur += 1
    else:
        perte_compteur = 0
        idx_lock = idx_local

    if perte_compteur >= perte_max:
        idx_lock = int(np.argmax(amp))
        perte_compteur = 0

    left_avg = max(0, idx_lock - 1)
    right_avg = min(n, idx_lock + 2)
    z_sel = np.mean(z_bin[left_avg:right_avg])

    return z_sel, idx_lock


# ============================================================
# Sélection HR robuste : init + tracking + réacquisition
# ============================================================
hr_prev_hz = np.nan
hr_last_valid_hz = np.nan
hr_last_valid_time = None

hold_max_s = 4.0
hr_delta_track = 0.18
hr_delta_reacq = 0.45
hr_beta = 0.25
hr_init_confirm_needed = 3

hr_init_history = deque(maxlen=5)


def est_proche_harmonique(freq, rr_hz, n_harm=5, tol=0.06):
    if not np.isfinite(freq) or not np.isfinite(rr_hz) or rr_hz <= 0:
        return False, None

    for n in range(2, n_harm + 1):
        fh = n * rr_hz
        if abs(freq - fh) < tol:
            return True, n

    return False, None


def score_candidat_hr(c, hr_prev, rr_hz):
    score = c["snr"]

    # on valorise les candidats non collés aux harmoniques RR
    rejet, n_h = est_proche_harmonique(c["freq"], rr_hz, n_harm=5, tol=0.06)
    if rejet:
        score -= 6.0

    # plage physiologique centrale plus probable
    if 0.95 <= c["freq"] <= 1.80:
        score += 1.5

    # proximité temporelle si suivi existant
    if np.isfinite(hr_prev):
        ecart = abs(c["freq"] - hr_prev)
        score -= 8.0 * ecart

    return score


def lisser_hr(hr_new, hr_old, beta=0.25):
    if not np.isfinite(hr_new):
        return hr_old
    if not np.isfinite(hr_old):
        return hr_new
    return (1 - beta) * hr_old + beta * hr_new


def choisir_hr_robuste(candidats_hr, rr_hz, t_now):
    global hr_prev_hz, hr_last_valid_hz, hr_last_valid_time, hr_init_history

    if len(candidats_hr) == 0:
        if hr_last_valid_time is not None and (t_now - hr_last_valid_time) <= hold_max_s:
            return hr_last_valid_hz
        return np.nan

    # on garde uniquement des candidats avec SNR minimal
    valides = [c for c in candidats_hr if c["snr"] >= 2.5]

    # si tout est faible, on garde quand même les 2 meilleurs pour ne pas bloquer trop tôt
    if len(valides) == 0:
        valides = candidats_hr[:2]

    # ----------------------------------------
    # Cas 1 : on a déjà un suivi valide
    # ----------------------------------------
    if np.isfinite(hr_prev_hz):
        proches = [c for c in valides if abs(c["freq"] - hr_prev_hz) <= hr_delta_track]

        if len(proches) > 0:
            best = max(proches, key=lambda c: score_candidat_hr(c, hr_prev_hz, rr_hz))
            hr_prev_hz = lisser_hr(best["freq"], hr_prev_hz, beta=hr_beta)
            hr_last_valid_hz = hr_prev_hz
            hr_last_valid_time = t_now
            return hr_prev_hz

        # pas de candidat proche : tentative de réacquisition contrôlée
        plausibles = []
        for c in valides:
            rejet, _ = est_proche_harmonique(c["freq"], rr_hz, n_harm=5, tol=0.06)
            if not rejet and 0.90 <= c["freq"] <= 2.00:
                plausibles.append(c)

        if len(plausibles) > 0:
            best = max(plausibles, key=lambda c: score_candidat_hr(c, hr_prev_hz, rr_hz))

            # réacquisition seulement si pas trop loin non plus
            if abs(best["freq"] - hr_prev_hz) <= hr_delta_reacq:
                hr_prev_hz = lisser_hr(best["freq"], hr_prev_hz, beta=hr_beta)
                hr_last_valid_hz = hr_prev_hz
                hr_last_valid_time = t_now
                return hr_prev_hz

        # sinon hold temporaire
        if hr_last_valid_time is not None and (t_now - hr_last_valid_time) <= hold_max_s:
            return hr_last_valid_hz

        hr_prev_hz = np.nan
        return np.nan

    # ----------------------------------------
    # Cas 2 : initialisation
    # ----------------------------------------
    plausibles = []
    for c in valides:
        rejet, _ = est_proche_harmonique(c["freq"], rr_hz, n_harm=5, tol=0.06)
        if not rejet and 0.90 <= c["freq"] <= 2.00:
            plausibles.append(c)

    if len(plausibles) == 0:
        if hr_last_valid_time is not None and (t_now - hr_last_valid_time) <= hold_max_s:
            return hr_last_valid_hz
        return np.nan

    best = max(plausibles, key=lambda c: score_candidat_hr(c, np.nan, rr_hz))
    hr_init_history.append(best["freq"])

    # confirmation sur plusieurs fenêtres
    if len(hr_init_history) >= hr_init_confirm_needed:
        recent = np.array(list(hr_init_history)[-hr_init_confirm_needed:], dtype=np.float64)
        if np.std(recent) < 0.12:
            hr_prev_hz = float(np.mean(recent))
            hr_last_valid_hz = hr_prev_hz
            hr_last_valid_time = t_now
            return hr_prev_hz

    if hr_last_valid_time is not None and (t_now - hr_last_valid_time) <= hold_max_s:
        return hr_last_valid_hz

    return np.nan


# ============================================================
# Buffers / paramètres
# ============================================================
fenetre_signal = 20.0
echantillons_min = 256
periode_affichage = 1.0
periode_traitement = 0.5

buffer_t = deque()
buffer_phi = deque()

fs_est = None
beta_fs = 0.1
last_print = time.perf_counter()
last_processing = time.perf_counter()

rr_last = np.nan
hr_last = np.nan


# ============================================================
# Boucle principale
# ============================================================
try:
    while True:
        code_erreur, resultat, tableau_IQ = uRAD_RP_SDK11.detection()
        if code_erreur != 0:
            close_program()

        I_brut = np.asarray(tableau_IQ[0], dtype=np.float64)
        Q_brut = np.asarray(tableau_IQ[1], dtype=np.float64)
        z_bin = I_brut + 1j * Q_brut

        z_sel, idx_sel = selection_bin_verrouillee(z_bin)

        I_sel = float(np.real(z_sel))
        Q_sel = float(np.imag(z_sel))

        I_c, Q_c = pretraitement_iq(I_sel, Q_sel)
        phase = np.arctan2(Q_c, I_c)

        t_now = time.perf_counter()
        buffer_t.append(t_now)
        buffer_phi.append(phase)

        if len(buffer_t) >= 2:
            dt = buffer_t[-1] - buffer_t[-2]
            if dt > 0:
                fs_inst = 1.0 / dt
                fs_est = fs_inst if fs_est is None else (1 - beta_fs) * fs_est + beta_fs * fs_inst

        while len(buffer_t) > 1 and (buffer_t[-1] - buffer_t[0]) > fenetre_signal:
            buffer_t.popleft()
            buffer_phi.popleft()

        if (
            fs_est is not None
            and len(buffer_phi) >= echantillons_min
            and (t_now - last_processing) >= periode_traitement
        ):
            phi = np.unwrap(np.array(buffer_phi, dtype=np.float64))
            phi_hp = filtre_passe_haut(phi, fs_est, fc=0.05)

            rr_hz = np.nan
            hr_hz = np.nan

            if fs_est > 1.2:
                sig_rr = filtre_passe_bande(phi_hp, 0.10, 0.50, fs_est)
                rr_hz = frequence_dominante(sig_rr, fs_est, 0.10, 0.50)

            if fs_est > 5.2:
                sig_hr = filtre_passe_bande(phi_hp, 0.80, 2.50, fs_est)
                hr_candidats = extraire_pics_spectraux(sig_hr, fs_est, 0.80, 2.50, max_peaks=6)
                hr_hz = choisir_hr_robuste(hr_candidats, rr_hz, t_now)

            if np.isfinite(rr_hz):
                rr_last = rr_hz
            if np.isfinite(hr_hz):
                hr_last = hr_hz

            rr_rpm = rr_last * 60.0 if np.isfinite(rr_last) else np.nan
            hr_bpm = hr_last * 60.0 if np.isfinite(hr_last) else np.nan

            if (t_now - last_print) >= periode_affichage:
                print(
                    f"bin={idx_sel:3d} | fs={fs_est:6.2f} Hz | "
                    f"RR={rr_last:5.3f} Hz ({rr_rpm:6.2f} rpm) | "
                    f"HR={hr_last:5.3f} Hz ({hr_bpm:6.2f} bpm)"
                )
                last_print = t_now

            last_processing = t_now

except KeyboardInterrupt:
    close_program()