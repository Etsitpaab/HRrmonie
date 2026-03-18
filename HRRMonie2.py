import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, get_window
import uRAD_RP_SDK11


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


# Initialisation radar
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
# Prétraitement IQ minimal : suppression DC
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

    I_c = I_in - mean_I
    Q_c = Q_in - mean_Q

    return I_c, Q_c


# ============================================================
# Filtres
# ============================================================
def filtre_passe_haut(signal, fs, fc=0.05, ordre=4):
    x = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0

    if len(x) < 16 or fc <= 0 or fc >= nyq:
        return x.copy()

    sos = butter(ordre, fc / nyq, btype='highpass', output='sos')
    return sosfiltfilt(sos, x)


def filtre_passe_bande(signal, f_low, f_high, fs, ordre=4):
    x = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0

    if len(x) < 16:
        return x.copy()

    if f_low <= 0 or f_high >= nyq or f_low >= f_high:
        raise ValueError("Bornes de bande invalides.")

    sos = butter(ordre, [f_low / nyq, f_high / nyq], btype='bandpass', output='sos')
    return sosfiltfilt(sos, x)


# ============================================================
# FFT + fréquence dominante
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
    f_band = f[mask]
    P_band = P[mask]

    if len(P_band) == 0:
        return np.nan

    idx = np.argmax(P_band)
    return float(f_band[idx])


# ============================================================
# Paramètres pipeline
# ============================================================
fenetre_signal = 30.0
echantillons_min = 200
periode_affichage = 1.0

buffer_t = []
buffer_phi = []

fs_est = None
beta_fs = 0.05
last_print = time.time()


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

        # Pipeline de base :
        # on prend simplement le bin de plus forte amplitude
        z_bin = I_brut + 1j * Q_brut
        idx_sel = int(np.argmax(np.abs(z_bin)))
        z_sel = z_bin[idx_sel]

        I_sel = np.real(z_sel)
        Q_sel = np.imag(z_sel)

        # Prétraitement IQ
        I_c, Q_c = pretraitement_iq(I_sel, Q_sel)

        # Estimation phase
        phase = np.arctan2(Q_c, I_c)

        # Temps
        t_now = time.time()
        buffer_t.append(t_now)
        buffer_phi.append(phase)

        # Estimation fréquence d'échantillonnage
        if len(buffer_t) >= 2:
            dt = buffer_t[-1] - buffer_t[-2]
            fs_inst = 1.0 / max(dt, 1e-6)

            if fs_est is None:
                fs_est = fs_inst
            else:
                fs_est = (1 - beta_fs) * fs_est + beta_fs * fs_inst

        # Fenêtre glissante
        while len(buffer_t) > 1 and (buffer_t[-1] - buffer_t[0]) > fenetre_signal:
            buffer_t.pop(0)
            buffer_phi.pop(0)

        # Traitement si assez d'échantillons
        if len(buffer_phi) >= echantillons_min and fs_est is not None:
            phi_wrap = np.array(buffer_phi, dtype=np.float64)

            # Dépliage de phase
            phi = np.unwrap(phi_wrap)

            # Suppression dérive lente
            phi_hp = filtre_passe_haut(phi, fs_est, fc=0.05)

            # Filtrage passe-bande
            sig_rr = filtre_passe_bande(phi_hp, 0.10, 0.50, fs_est)
            sig_hr = filtre_passe_bande(phi_hp, 0.80, 2.50, fs_est)

            # Fréquences dominantes
            rr_hz = frequence_dominante(sig_rr, fs_est, 0.10, 0.50)
            hr_hz = frequence_dominante(sig_hr, fs_est, 0.80, 2.50)

            rr_rpm = rr_hz * 60.0 if np.isfinite(rr_hz) else np.nan
            hr_bpm = hr_hz * 60.0 if np.isfinite(hr_hz) else np.nan

            # Affichage simple
            if (t_now - last_print) > periode_affichage:
                print(
                    f"bin = {idx_sel:3d} | "
                    f"fs = {fs_est:6.2f} Hz | "
                    f"RR = {rr_hz:5.3f} Hz ({rr_rpm:6.2f} rpm) | "
                    f"HR = {hr_hz:5.3f} Hz ({hr_bpm:6.2f} bpm)"
                )
                last_print = t_now

except KeyboardInterrupt:
    close_program()