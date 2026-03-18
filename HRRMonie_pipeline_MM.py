import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, get_window
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
# FFT
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
# Motion gating robuste avec hystérésis
# ============================================================
etat_stable = False
stable_count = 0
unstable_count = 0

# nombre de validations consécutives avant changement d'état
stable_needed = 4
unstable_needed = 6


def metriques_stabilite(phi_hp):
    phi_hp = np.asarray(phi_hp, dtype=np.float64)

    if len(phi_hp) < 20:
        return np.inf, np.inf

    dphi = np.diff(phi_hp)
    return float(np.std(phi_hp)), float(np.std(dphi))


def maj_etat_stabilite(phi_hp):
    global etat_stable, stable_count, unstable_count

    std_phi, std_dphi = metriques_stabilite(phi_hp)

    # Hystérésis :
    # - pour devenir stable : seuils stricts
    # - pour rester stable : seuils plus permissifs
    if not etat_stable:
        cond_stable = (std_phi < 1.00) and (std_dphi < 0.18)
        if cond_stable:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= stable_needed:
            etat_stable = True
            unstable_count = 0
    else:
        cond_instable = (std_phi > 1.35) or (std_dphi > 0.30)
        if cond_instable:
            unstable_count += 1
        else:
            unstable_count = 0

        if unstable_count >= unstable_needed:
            etat_stable = False
            stable_count = 0

    return etat_stable, std_phi, std_dphi


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

        # Bin thorax stabilisé
        z_sel, idx_sel = selection_bin_verrouillee(z_bin)

        I_sel = float(np.real(z_sel))
        Q_sel = float(np.imag(z_sel))

        # Prétraitement IQ
        I_c, Q_c = pretraitement_iq(I_sel, Q_sel)

        # Phase
        phase = np.arctan2(Q_c, I_c)

        t_now = time.perf_counter()
        buffer_t.append(t_now)
        buffer_phi.append(phase)

        # Estimation fs
        if len(buffer_t) >= 2:
            dt = buffer_t[-1] - buffer_t[-2]
            if dt > 0:
                fs_inst = 1.0 / dt
                fs_est = fs_inst if fs_est is None else (1 - beta_fs) * fs_est + beta_fs * fs_inst

        # Fenêtre glissante
        while len(buffer_t) > 1 and (buffer_t[-1] - buffer_t[0]) > fenetre_signal:
            buffer_t.popleft()
            buffer_phi.popleft()

        # Traitement périodique
        if (
            fs_est is not None
            and len(buffer_phi) >= echantillons_min
            and (t_now - last_processing) >= periode_traitement
        ):
            phi = np.unwrap(np.array(buffer_phi, dtype=np.float64))
            phi_hp = filtre_passe_haut(phi, fs_est, fc=0.05)

            stable, std_phi, std_dphi = maj_etat_stabilite(phi_hp)

            if stable:
                rr_hz = np.nan
                hr_hz = np.nan

                if fs_est > 1.2:
                    sig_rr = filtre_passe_bande(phi_hp, 0.10, 0.50, fs_est)
                    rr_hz = frequence_dominante(sig_rr, fs_est, 0.10, 0.50)

                if fs_est > 5.2:
                    sig_hr = filtre_passe_bande(phi_hp, 0.80, 2.50, fs_est)
                    hr_hz = frequence_dominante(sig_hr, fs_est, 0.80, 2.50)

                if np.isfinite(rr_hz):
                    rr_last = rr_hz
                if np.isfinite(hr_hz):
                    hr_last = hr_hz

            rr_rpm = rr_last * 60.0 if np.isfinite(rr_last) else np.nan
            hr_bpm = hr_last * 60.0 if np.isfinite(hr_last) else np.nan

            if (t_now - last_print) >= periode_affichage:
                print(
                    f"bin={idx_sel:3d} | fs={fs_est:6.2f} Hz | "
                    f"etat={'stable' if stable else 'mouvement'} | "
                    f"std_phi={std_phi:5.3f} | std_dphi={std_dphi:5.3f} | "
                    f"RR={rr_last:5.3f} Hz ({rr_rpm:6.2f} rpm) | "
                    f"HR={hr_last:5.3f} Hz ({hr_bpm:6.2f} bpm)"
                )
                last_print = t_now

            last_processing = t_now

except KeyboardInterrupt:
    close_program()