import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, get_window
from collections import deque
import uRAD_RP_SDK11

# =========================
# Configuration radar uRAD
# =========================
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

# =========================
# Paramètres respiration
# =========================
# Référence test: ~15 rpm => 0.25 Hz
# Bande resserrée pour éviter les dérives trop lentes
RR_MIN_HZ = 0.20          # 12 rpm
RR_MAX_HZ = 0.32          # 19.2 rpm

RR_TRACK_JUMP_RPM = 3.0   # variation max entre deux sorties stables
RR_SNR_MIN = 3.0
RR_PROM_MIN = 0.02

HP_FC_HZ = 0.08           # suppression dérive lente plus ferme
RR_WINDOW_SEC = 30.0      # fenêtre plus robuste que 20 s
DISPLAY_PERIOD = 1.0

# Validation croisée FFT / autocorrélation
RR_FFT_ACCORD_MAX_RPM = 2.0

# Lissage sortie
RR_EMA_ALPHA = 0.20
RR_CONFIRM_COUNT = 2      # nb de fenêtres cohérentes avant MAJ sortie

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
# Prétraitement phase
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
def filtre_pass_haut(signal, fs, fc=0.08):
    signal = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0
    if fc >= nyq:
        return signal.copy()
    sos = butter(4, fc / nyq, btype='highpass', output='sos')
    return sosfiltfilt(sos, signal)

def passe_bande(signal, f_basse, f_haut, fs, ordre=4):
    signal = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0

    if f_basse <= 0 or f_haut >= nyq or f_basse >= f_haut:
        raise ValueError("Bornes de passe-bande invalides.")

    sos = butter(ordre, [f_basse / nyq, f_haut / nyq],
                 btype='bandpass', output='sos')
    return sosfiltfilt(sos, signal)

# =========================
# Spectre / FFT
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
        snr_db = 10.0 * np.log10(pic_pow / bruit)
        return frequence_pic, snr_db, 0.0

    idx_best = np.argmax(Pxx_bande[pics])
    k0 = pics[idx_best]
    frequence_pic = frequence_bande[k0]
    pic_pow = Pxx_bande[k0] + 1e-12
    snr_db = 10.0 * np.log10(pic_pow / bruit)

    prominences = propriete["prominences"]
    prom = float(prominences[idx_best])
    prominence_norm = prom / pic_pow

    return frequence_pic, snr_db, prominence_norm

def estimation_rr_fft(signal_rr, fs):
    freqs, Pxx = puissance_spectre(signal_rr, fs, nfft=4 * len(signal_rr), window="hann")
    f_pic, snr_db, prom = qualite_pics(freqs, Pxx, RR_MIN_HZ, RR_MAX_HZ)
    return f_pic * 60.0 if np.isfinite(f_pic) else np.nan, snr_db, prom

# =========================
# Autocorrélation
# =========================
def estimation_rr_autocorr(signal_rr, fs, fmin=RR_MIN_HZ, fmax=RR_MAX_HZ):
    x = np.asarray(signal_rr, dtype=np.float64)
    x = x - np.mean(x)

    if np.allclose(x, 0):
        return np.nan

    ac = np.correlate(x, x, mode='full')
    ac = ac[len(ac)//2:]

    lag_min = int(fs / fmax)
    lag_max = int(fs / fmin)

    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, len(ac) - 1)

    if lag_max <= lag_min:
        return np.nan

    zone = ac[lag_min:lag_max + 1]
    if zone.size == 0:
        return np.nan

    idx = np.argmax(zone)
    lag_best = lag_min + idx
    f_est = fs / lag_best
    return f_est * 60.0

# =========================
# Validation physiologique
# =========================
rr_last_output = np.nan
rr_candidate_prev = np.nan
rr_confirm_counter = 0

def valider_rr(rr_fft, rr_ac, snr_db, prom):
    global rr_last_output, rr_candidate_prev, rr_confirm_counter

    # 1) Conditions qualité FFT
    if not np.isfinite(rr_fft):
        return rr_last_output
    if not np.isfinite(snr_db) or snr_db < RR_SNR_MIN:
        return rr_last_output
    if not np.isfinite(prom) or prom < RR_PROM_MIN:
        return rr_last_output

    # 2) Accord FFT / autocorr
    if not np.isfinite(rr_ac):
        return rr_last_output
    if abs(rr_fft - rr_ac) > RR_FFT_ACCORD_MAX_RPM:
        return rr_last_output

    # 3) Candidat fusionné
    rr_candidate = 0.5 * (rr_fft + rr_ac)

    # 4) Confirmation sur plusieurs fenêtres
    if np.isfinite(rr_candidate_prev) and abs(rr_candidate - rr_candidate_prev) <= 1.5:
        rr_confirm_counter += 1
    else:
        rr_confirm_counter = 1
    rr_candidate_prev = rr_candidate

    if rr_confirm_counter < RR_CONFIRM_COUNT:
        return rr_last_output

    # 5) Continuité physiologique
    if np.isfinite(rr_last_output):
        if abs(rr_candidate - rr_last_output) > RR_TRACK_JUMP_RPM:
            return rr_last_output
        rr_last_output = (1 - RR_EMA_ALPHA) * rr_last_output + RR_EMA_ALPHA * rr_candidate
    else:
        rr_last_output = rr_candidate

    return rr_last_output

# =========================
# Buffers temporels
# =========================
buffer_temps = deque()
buffer_phi = deque()

seconde_fenetre = 40.0       # on garde un peu plus que la fenêtre d'analyse
echantillon_min = 250
fs_est = None
beta_fs = 0.05
marqueur_print = time.time()

# =========================
# Boucle principale
# =========================
try:
    while True:
        code_erreur, resultat, tableau_IQ = uRAD_RP_SDK11.detection()
        if code_erreur != 0:
            closeProgram()

        # En mode CW : pas de range bin réel
        # On prend une mesure complexe globale stable
        I_brut = np.asarray(tableau_IQ[0], dtype=np.float64)
        Q_brut = np.asarray(tableau_IQ[1], dtype=np.float64)
        z = I_brut + 1j * Q_brut
        z_sel = np.mean(z)

        I_sel = float(np.real(z_sel))
        Q_sel = float(np.imag(z_sel))

        # Phase physiologique
        I_phase, Q_phase = pretraitement_phase(I_sel, Q_sel)
        phase = np.arctan2(Q_phase, I_phase)
        phase_deplie_val = unwrap_phase(phase)

        # Buffer temporel
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

        # Traitement respiration
        if len(buffer_phi) >= echantillon_min:
            tab_phi = np.array(buffer_phi, dtype=np.float64)

            if fs_est is not None:
                fs = fs_est
            else:
                tab_t = np.array(buffer_temps, dtype=np.float64)
                dt = np.diff(tab_t)
                fs = 1.0 / np.median(dt)

            # 1) suppression dérive lente
            phi_hp = filtre_pass_haut(tab_phi, fs, fc=HP_FC_HZ)

            # 2) bande respiration
            try:
                sig_rr = passe_bande(phi_hp, RR_MIN_HZ, RR_MAX_HZ, fs)
            except ValueError:
                continue

            # 3) FFT
            rr_fft, rr_snr, rr_prom = estimation_rr_fft(sig_rr, fs)

            # 4) Autocorr
            rr_ac = estimation_rr_autocorr(sig_rr, fs)

            # 5) Validation / lissage
            rr_out = valider_rr(rr_fft, rr_ac, rr_snr, rr_prom)

            if (t_now - marqueur_print) > DISPLAY_PERIOD:
                print(
                    f"RR_affichee: {rr_out:.2f} rpm | "
                    f"RR_fft: {rr_fft:.2f} rpm | "
                    f"RR_ac: {rr_ac:.2f} rpm | "
                    f"SNR_RR: {rr_snr:.2f} dB | "
                    f"PROM_RR: {rr_prom:.4f} | "
                    f"fs: {fs:.2f} Hz"
                )
                marqueur_print = t_now

except KeyboardInterrupt:
    closeProgram()