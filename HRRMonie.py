import time
import math
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, get_window
import uRAD_RP_SDK11
import matplotlib.pyplot as plt
from collections import deque

# -------------------------
# Configuration radar uRAD
# -------------------------
mode = 1
f0 = 125
BW = 240
Ns = 200
Ntar = 1 # 1 à 5
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

#Code recommandé par Urad
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

# Fonction du pipeline de traitement du signal radar

beta_dc = 0.001
beta_variance = 0.001
epsilon = 1e-6

moyenne_I = None
moyenne_Q = None
variance_I = None
variance_Q = None


def pretraitement(I_entree,Q_entree): # Etape 2 du pipeline

    global moyenne_I, moyenne_Q, variance_I, variance_Q

    I_entree = float(I_entree)
    Q_entree = float(Q_entree)

    if moyenne_I is None:
        moyenne_I, moyenne_Q = I_entree, Q_entree
        variance_I, variance_Q = 1.0, 1.0

    moyenne_I = (1 - beta_dc) * moyenne_I + beta_dc * I_entree
    moyenne_Q = (1 - beta_dc) * moyenne_Q + beta_dc * Q_entree
    I_centre = I_entree - moyenne_I
    Q_centre = Q_entree - moyenne_Q

    variance_I = (1 - beta_variance) * variance_I + beta_variance * (I_centre ** 2)
    variance_Q = (1 - beta_variance) * variance_Q + beta_variance * (Q_centre ** 2)

    I_normalise = I_centre / np.sqrt(variance_I + epsilon)
    Q_normalise = Q_centre / np.sqrt(variance_Q + epsilon)

    return I_normalise, Q_normalise

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
        delta -= 2*np.pi
    elif delta < -np.pi:
        delta += 2*np.pi

    phase_deplie += delta
    phase_precedente = phase_actuelle
    return phase_deplie

def filtre_pass_haut(phase_signal, freq_echanti): # Etape 3 
    nyq = freq_echanti / 2

    if 0.05 >= nyq:
        return phase_signal
    
    sos = butter(4, 0.05 / nyq, btype='highpass', output='sos')

    signal_filtre = sosfiltfilt(sos,phase_signal)

    return signal_filtre

def passe_bande(phase_signal, f_basse, f_haut, freq_echanti): # Etape 4
    ordre = 4

    signal = np.asarray(phase_signal, dtype=np.float64)

    nyq = freq_echanti / 2

    if f_haut >= nyq:
        raise ValueError("La fréquence de coupure haute doit être inférieure à la moitié de la fréquence d'échantillonnage.")
    sos=butter(ordre, [f_basse /nyq, f_haut / nyq], 
               btype='bandpass', output='sos')

    signal_filtre = sosfiltfilt(sos, signal)

    return signal_filtre

def puissance_spectre(x, fs, nfft=None, window="hann"):
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(max(N, 1))))
    w = get_window(window, N)
    xw = (x - np.mean(x)) * w
    X = np.fft.rfft(xw, n=nfft)
    Pxx = (np.abs(X) ** 2)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return f, Pxx


def qualite_pics(frequence, Pxx, fmin, fmax):
    bande = (frequence >= fmin) & (frequence <= fmax)
    frequence_bande = frequence[bande]
    Pxx_bande = Pxx[bande]

    if Pxx_bande.size == 0:
        return np.nan, np.nan, 0.0

    pics, propriete = find_peaks(Pxx_bande, prominence=0.1 * np.max(Pxx_bande))

    bruit = np.median(Pxx_bande) + 1e-12

    if len(pics) == 0:
        k = int(np.argmax(Pxx_bande))
        frequence_pic = frequence_bande[k]
        pic_pow = Pxx_bande[k] + 1e-12
        SNR_db = 10.0 * np.log10(pic_pow / bruit)
        return frequence_pic, SNR_db, 0.0

    k0 = pics[np.argmax(Pxx_bande[pics])]
    frequence_pic = frequence_bande[k0]
    pic_pow = Pxx_bande[k0] + 1e-12
    SNR_db = 10.0 * np.log10(pic_pow / bruit)

    prominences = propriete["prominences"]
    prom = float(prominences[np.argmax(Pxx_bande[pics])])
    prominence_norm = prom / pic_pow

    return frequence_pic, SNR_db, prominence_norm


def FFT_glissante(x, freq, freq_min, freq_max, duree_fenetre=20.0):
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    fen = int(round(duree_fenetre * freq))
    if fen < 5:
        raise ValueError("La durée de la fenêtre doit être suffisante pour contenir au moins 5 échantillons.")
    
    saut = int(round(fen * (1.0 * 0.5)))
    hop = max(1, saut)

    temps_centre = []
    frequence_pic = []
    decibel_SNR = []
    prominence = []

    i = 0
    while i + fen <= N:
        segment = x[i:i+fen]
        freqs, Pxx = puissance_spectre(segment, freq, nfft=fen, window="hann")
        f_pic, SNR_db, prom = qualite_pics(freqs, Pxx, freq_min, freq_max)
        temps_centre.append((i + fen / 2) / freq)
        frequence_pic.append(f_pic)
        decibel_SNR.append(SNR_db)
        prominence.append(prom)
        i += hop
    return np.array(temps_centre), np.array(frequence_pic), np.array(decibel_SNR), np.array(prominence)


def tracking_freq(freq_est, snr_db, prom_norm,saut_max=0.15, alph=0.3, snr_min=3.0, prom_min=0.02):


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


def estimation_rr(signal_rr, fs, duree_fenetre=20.0):
    overlap = 0.5
    t,f,snr,prom = FFT_glissante(signal_rr, fs, freq_min=0.10, freq_max=0.50, duree_fenetre=duree_fenetre)
    f_tr = tracking_freq(f, snr, prom,saut_max=0.05, alph=0.25)
    rpm = f_tr * 60.0
    return t, rpm, snr, prom

def estimation_hr(signal_hr,fs, duree_fenetre=15.0):
    t,f,snr,prom = FFT_glissante(signal_hr, fs, freq_min=0.8, freq_max=2.50, duree_fenetre=duree_fenetre)
    f_tr = tracking_freq(f, snr, prom, saut_max=0.2, alph=0.3)

    bpm = f_tr * 60.0
    
    print(f"f HR détectée: {f_tr[~np.isnan(f_tr)][-1] if np.any(~np.isnan(f_tr)) else np.nan}")
    return t, bpm, snr, prom


#Boucle infinie d'execution du radar
plt.ion()  # mode interactif

fig, ax = plt.subplots()
scatter = ax.scatter([], [], s=8)

ax.set_xlabel("I")
ax.set_ylabel("Q")
ax.set_title("Constellation I/Q")
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

# bornes fixes (important pour éviter le rescale permanent)
ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)



seconde_fenetre = 30.0
echantillon_min = 200
affichage = 1.0


buffer_temps = deque()
buffer_phi = deque()

marqueur_print = time.time()

fs_est = None
beta_fs = 0.05

try:
    while True:
        code_erreur, resultat, tableau_IQ = uRAD_RP_SDK11.detection()
        if code_erreur != 0:
            closeProgram()
            break
        
        # -----------------------
        # Pipeline étape 1
        # -----------------------
        I_brut = np.asarray(tableau_IQ[0],dtype=np.float64)
        Q_brut = np.asarray(tableau_IQ[1],dtype=np.float64)

        z_bin = I_brut + 1j * Q_brut
        
        idx = int(np.argmax(np.abs(z_bin)))
        
        I_moyenne = float(I_brut[idx])
        Q_moyenne = float(Q_brut[idx])
        I_n, Q_n = pretraitement(I_moyenne, Q_moyenne)


        #Graphe de constellation :
        x = np.array(I_n) + 1j*np.array(Q_n)
        x = x - np.mean(x)  # suppression offset DC

        # mise à jour des points
        plt.scatter.set_offsets(np.c_[np.real(x), np.imag(x)])

        plt.fig.canvas.draw()
        plt.fig.canvas.flush_events()
        plt.pause(0.001)  # très court, ne bloque pas

        phase = np.arctan2(Q_n, I_n)
        phase_deplie_val = unwrap_phase(phase)

        t_now = time.time()
        buffer_temps.append(t_now)
        buffer_phi.append(phase_deplie_val)

        # -----------------------
        # Estimation fs lissée (EMA)
        # -----------------------
        if len(buffer_temps) >= 2:
            dt_last = buffer_temps[-1] - buffer_temps[-2]
            fs_inst = 1.0 / max(dt_last, 1e-6)
            fs_est = fs_inst if fs_est is None else (1 - beta_fs) * fs_est + beta_fs * fs_inst

        # -----------------------
        # Fenêtre glissante temporelle
        # -----------------------
        while (buffer_temps[-1] - buffer_temps[0]) > seconde_fenetre:
            buffer_temps.popleft()
            buffer_phi.popleft()
        
        # -----------------------
        # Traitement si assez d’échantillons
        # -----------------------
        if len(buffer_phi) >= echantillon_min:

            tab_phi = np.array(buffer_phi, dtype=np.float64)

            # On utilise fs lissé si dispo
            if fs_est is not None:
                fs = fs_est
            else:
                tab_t = np.array(buffer_temps, dtype=np.float64)
                dt = np.diff(tab_t)
                fs = 1.0 / np.median(dt)

            phi_passe_haut = filtre_pass_haut(tab_phi, fs)

            sig_rr = passe_bande(phi_passe_haut, 0.10, 0.50, fs)
            sig_hr = passe_bande(phi_passe_haut, 1.20, 2.20, fs)

            if (t_now - marqueur_print) > affichage:

                t_rr, rr_rpm, rr_snr, rr_prom = estimation_rr(sig_rr, fs, duree_fenetre=20.0)
                rr_val = rr_rpm[~np.isnan(rr_rpm)]
                rr_out = rr_val[-1] if rr_val.size else np.nan

                t_hr, hr_bpm, hr_snr, hr_prom = estimation_hr(sig_hr, fs, duree_fenetre=15.0)
                hr_val = hr_bpm[~np.isnan(hr_bpm)]
                hr_out = hr_val[-1] if hr_val.size else np.nan

                print(f"RR: {rr_out:.2f} rpm, HR: {hr_out:.2f} bpm")
                
                marqueur_print = t_now

except KeyboardInterrupt:
    closeProgram()