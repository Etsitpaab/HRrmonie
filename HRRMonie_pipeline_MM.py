import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, get_window
import uRAD_RP_SDK11
import matplotlib.pyplot as plt
from collections import deque


# Configuration radar uRAD

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



# 1) Prétraitement IQ 

beta_dc_iq = 0.001
beta_var_iq = 0.001
eps = 1e-6

mean_I_iq = None
mean_Q_iq = None
var_I_iq = None
var_Q_iq = None


def pretraitement_affichage_iq(I_in, Q_in):
    global mean_I_iq, mean_Q_iq, var_I_iq, var_Q_iq

    I_in = float(I_in)
    Q_in = float(Q_in)

    if mean_I_iq is None:
        mean_I_iq = I_in
        mean_Q_iq = Q_in
        var_I_iq = 1.0
        var_Q_iq = 1.0

    mean_I_iq = (1 - beta_dc_iq) * mean_I_iq + beta_dc_iq * I_in
    mean_Q_iq = (1 - beta_dc_iq) * mean_Q_iq + beta_dc_iq * Q_in

    I_c = I_in - mean_I_iq
    Q_c = Q_in - mean_Q_iq

    var_I_iq = (1 - beta_var_iq) * var_I_iq + beta_var_iq * (I_c ** 2)
    var_Q_iq = (1 - beta_var_iq) * var_Q_iq + beta_var_iq * (Q_c ** 2)

    I_n = I_c / np.sqrt(var_I_iq + eps)
    Q_n = Q_c / np.sqrt(var_Q_iq + eps)

    return I_n, Q_n



# 2) Prétraitement pour phase

beta_dc_phase = 0.01
mean_I_phase = None
mean_Q_phase = None


def pretraitement_phase(I_in, Q_in):
    global mean_I_phase, mean_Q_phase

    I_in = float(I_in)
    Q_in = float(Q_in)

    if mean_I_phase is None:
        mean_I_phase = I_in
        mean_Q_phase = Q_in

    mean_I_phase = (1 - beta_dc_phase) * mean_I_phase + beta_dc_phase * I_in
    mean_Q_phase = (1 - beta_dc_phase) * mean_Q_phase + beta_dc_phase * Q_in

    I_c = I_in - mean_I_phase
    Q_c = Q_in - mean_Q_phase

    return I_c, Q_c

# 3) Filtres

def filtre_pass_haut(signal, fs, fc=0.05, ordre=4):
    x = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0
    if fc <= 0 or fc >= nyq:
        return x.copy()
    sos = butter(ordre, fc / nyq, btype='highpass', output='sos')
    return sosfiltfilt(sos, x)


def passe_bande(signal, f_low, f_high, fs, ordre=4):
    x = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0
    if f_low <= 0 or f_high >= nyq or f_low >= f_high:
        raise ValueError("Bornes de bande invalides.")
    sos = butter(ordre, [f_low / nyq, f_high / nyq], btype='bandpass', output='sos')
    return sosfiltfilt(sos, x)



# 4) Estimation fréquentielle simple par FFT

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


def frequence_dominante(x, fs, fmin, fmax):
    f, P = spectre_fft(x, fs)

    if f.size == 0:
        return np.nan

    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return np.nan

    f_band = f[mask]
    P_band = P[mask]

    idx = int(np.argmax(P_band))
    return float(f_band[idx])


# 5) Affichage constellation

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


# 6) Buffers / paramètres

fenetre_signal = 30.0
affichage = 1.0
echantillons_min = 200

buffer_t = deque()
buffer_phi = deque()

last_print = time.time()
fs_est = None
beta_fs = 0.05


# 7) Sélection simple de bin

def selection_bin_simple(z_bin):
    idx = int(np.argmax(np.abs(z_bin)))
    left = max(0, idx - 1)
    right = min(len(z_bin), idx + 2)
    return np.mean(z_bin[left:right]), idx

# 8) Boucle principale

try:
    while True:
        code_erreur, resultat, tableau_IQ = uRAD_RP_SDK11.detection()
        if code_erreur != 0:
            closeProgram()

        I_brut = np.asarray(tableau_IQ[0], dtype=np.float64)
        Q_brut = np.asarray(tableau_IQ[1], dtype=np.float64)
        z_bin = I_brut + 1j * Q_brut

        
        # Sélection bin simple
        
        z_sel, idx_sel = selection_bin_simple(z_bin)
        I_sel = float(np.real(z_sel))
        Q_sel = float(np.imag(z_sel))

        
        # Constellation
        
        I_aff, Q_aff = pretraitement_affichage_iq(I_sel, Q_sel)
        z_aff = I_aff + 1j * Q_aff

        if np.abs(z_aff) > 0:
            z_aff = z_aff / np.abs(z_aff)

        histo_i.append(float(np.real(z_aff)))
        histo_q.append(float(np.imag(z_aff)))

        points = np.c_[histo_i, histo_q]
        sc.set_offsets(points)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

        
        # Pipeline phase BASIQUE
        
        I_c, Q_c = pretraitement_phase(I_sel, Q_sel)
        phase = np.arctan2(Q_c, I_c)

        t_now = time.time()
        buffer_t.append(t_now)
        buffer_phi.append(phase)

        # estimation fs
        if len(buffer_t) >= 2:
            dt = buffer_t[-1] - buffer_t[-2]
            fs_inst = 1.0 / max(dt, 1e-6)
            if fs_est is None:
                fs_est = fs_inst
            else:
                fs_est = (1 - beta_fs) * fs_est + beta_fs * fs_inst

        # fenêtre glissante
        while len(buffer_t) > 1 and (buffer_t[-1] - buffer_t[0]) > fenetre_signal:
            buffer_t.popleft()
            buffer_phi.popleft()
        
        # Traitement fréquentiel simple
        if len(buffer_phi) >= echantillons_min and fs_est is not None:
            phi_wrap = np.array(buffer_phi, dtype=np.float64)
            phi = np.unwrap(phi_wrap)

            # suppression dérive lente / composante DC
            phi_hp = filtre_pass_haut(phi, fs_est, fc=0.05)

            # bandes physiologiques de base
            sig_rr = passe_bande(phi_hp, 0.10, 0.50, fs_est)
            sig_hr = passe_bande(phi_hp, 0.80, 2.50, fs_est)

            rr_hz = frequence_dominante(sig_rr, fs_est, 0.10, 0.50)
            hr_hz = frequence_dominante(sig_hr, fs_est, 0.80, 2.50)

            rr_rpm = rr_hz * 60.0 if np.isfinite(rr_hz) else np.nan
            hr_bpm = hr_hz * 60.0 if np.isfinite(hr_hz) else np.nan

            if (t_now - last_print) > affichage:
                print(
                    f"idx={idx_sel:3d} | RR: {rr_rpm:.2f} rpm | HR: {hr_bpm:.2f} bpm | fs: {fs_est:.2f} Hz"
                )
                last_print = t_now

except KeyboardInterrupt:
    closeProgram()