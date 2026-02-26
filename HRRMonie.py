import time
import math
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, get_window
import uRAD_RP_SDK11
import matplotlib.pyplot as plt
import props
from collections import deque
# -------------------------
# Configuration radar uRAD
# -------------------------
mode = 1
f0 = 125
BW = 240
Ns = 200
Ntar = 3 # 1 à 5
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

beta_dc = 0.005
beta_variance = 0.01
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


def unwrap_phase(phase_actuel): #Etape 2
    global phase_deplie, phase_precedente

    if phase_precedente is None:
        phase_precedente = phase_actuel
        phase_deplie = phase_actuel
        return phase_deplie
    
    delta = phase_actuel - phase_precedente

    if delta > np.pi:
        phase_deplie -= 2*np.pi
    elif delta < -np.pi:
        phase_deplie += 2*np.pi

    phase_deplie +=delta
    phase_precedente = phase_actuel
    return phase_deplie

def filtre_pass_haut(phase_signal, freq_echanti): # Etape 3 
    nyq = freq_echanti / 2

    if 0.05 >= nyq:
        return phase_signal
    
    sos = butter(4, 0.05 / nyq, btype='highpass', output='sos')

    signal_filtre = sosfiltfilt(sos,phase_signal)

    return signal_filtre

def passe_bande(phase_signal, f_basse, f_haut, freq_echanti): # Etape 4
    #Formule générique
    ordre = 4

    signal = np.asarray(phase_signal, dtype=np.float64)

    nyq = freq_echanti / 2

    if f_haut >= nyq:
        raise ValueError("La fréquence de coupure haute doit être inférieure à la moitié de la fréquence d'échantillonnage.")
    sos=butter(ordre, [f_basse /nyq, f_haut / nyq], 
               btype='bandpass', output='sos')

    signal_filtre = sosfiltfilt(sos, signal)

    return signal_filtre

def puissance_spectre(x,freq,nfft=None, window="hann"):
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if nfft is None:
        nfft = int(2 - np.ceil(np.log2(N)))
    w = get_window(window, N)
    fenetre_x = (x - np.mean(x)) * w
    X = np.fft.rfft(fenetre_x, n=nfft)
    Pxx = np.abs(X)**2
    frequence = np.fft.rfftfreq(nfft, d=1.0/freq)

    return frequence, Pxx

def qualite_pics(frequence, Pxx, min, max):
    bande = (frequence >= min) & (frequence <= max)
    frequence_bande = frequence[bande]
    Pxx_bande = Pxx[bande]

    pics, propriete = find_peaks(Pxx_bande, prominence=0.1 * np.max(Pxx_bande))

    if len(pics) == 0:
        k = int(np.argmax(Pxx_bande))
        frequence_pic = frequence_bande[k]
        pic_pow = Pxx_bande[k]
        bruit = np.median(Pxx_bande)
        SNR_db = 10 * np.log10(pic_pow + 1e-12)/ (bruit + 1e-12)

        return frequence_pic, SNR_db, 0.0
    
    k0 = pics[np.argmax(Pxx_bande[pics])]
    frequence_pic = frequence_bande[k0]
    pic_pow = Pxx_bande[k0]
    bruit = np.median(Pxx_bande)
    SNR_db = 10 * np.log10(pic_pow + 1e-12) / (bruit + 1e-12)
    prominences = props["prominences"]
    prom = prominences[np.argmax(Pxx_bande[pics])]
    prominence_norm = float(prom / (pic_pow + 1e-12))

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
    return t, bpm, snr, prom


#Boucle infinie d'éxecution du radar
seconde_fenetre = 30.0
echantillon_min = 200
affichage = 1.0


buffer_temps = deque()
buffer_phi = deque()

marqueur_print = time.time()

try:
    while True:
        code_erreur, resultat, tableau_IQ = uRAD_RP_SDK11.detection()
        if code_erreur != 0:
            closeProgram()
            break
        
        # Pipeline etape 1
        I_brut = tableau_IQ[0]
        Q_brut = tableau_IQ[1]

        I_moyenne = np.mean(I_brut)
        Q_moyenne = np.mean(Q_brut)

        z = I_moyenne + 1j * Q_moyenne

        I_n, Q_n = pretraitement(I_moyenne, Q_moyenne) 

        phase = np.arctan2(Q_n, I_n)
        phase_deplie = unwrap_phase(phase)

        t_now = time.time()
        buffer_temps.append(t_now)
        buffer_phi.append(phase_deplie)

        while (buffer_temps[-1] - buffer_temps[0]) > seconde_fenetre:
            buffer_temps.popleft()
            buffer_phi.popleft()
        
        if (len(buffer_phi)>= echantillon_min):
            tab_t = np.array(buffer_temps, dtype=np.float64)
            tab_phi = np.array(buffer_phi, dtype=np.float64)

            dt = np.diff(tab_t)
            fs = 1.0 / np.median(dt)
            phi_passe_haut = filtre_pass_haut(tab_phi, fs)

            sig_rr = passe_bande(phi_passe_haut, 0.10, 0.50, fs)
            sig_hr = passe_bande(phi_passe_haut, 0.80,2.50, fs)

            if(t_now - marqueur_print ) > affichage: 
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
