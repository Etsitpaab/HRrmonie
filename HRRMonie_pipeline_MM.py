import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, get_window, find_peaks
import uRAD_RP_SDK11
import matplotlib.pyplot as plt
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


# ============================================================
# 1) Prétraitement IQ
# ============================================================
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


# ============================================================
# 2) Prétraitement pour phase
# ============================================================
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


# ============================================================
# 3) Filtres
# ============================================================
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


# ============================================================
# 4) Estimation fréquentielle FFT
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


def extraire_pics_spectraux(signal, fs, fmin, fmax, max_peaks=8):
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

    peaks_sorted = peaks[np.argsort(P_band[peaks])[::-1]]

    candidats = []
    for k in peaks_sorted[:max_peaks]:
        freq = float(f_band[k])
        power = float(P_band[k])
        snr = float(10.0 * np.log10((power + 1e-12) / bruit))
        candidats.append({
            "freq": freq,
            "power": power,
            "snr": snr
        })

    return candidats


def analyse_spectrale_debug(signal, fs, fmin, fmax):
    candidats = extraire_pics_spectraux(signal, fs, fmin, fmax, max_peaks=8)

    if len(candidats) == 0:
        return np.nan, np.nan, np.nan, np.nan, []

    f1 = candidats[0]["freq"]
    p1 = candidats[0]["power"]
    snr1 = candidats[0]["snr"]
    f2 = candidats[1]["freq"] if len(candidats) > 1 else np.nan

    return f1, f2, snr1, p1, candidats


# ============================================================
# 5) Choix HR robuste
# ============================================================
def est_proche_harmonique(freq, rr_hz, n_harm=5, tol=0.06):
    if not np.isfinite(freq) or not np.isfinite(rr_hz) or rr_hz <= 0:
        return False, None

    for n in range(2, n_harm + 1):
        fh = n * rr_hz
        if abs(freq - fh) < tol:
            return True, n
    return False, None


def choisir_hr_robuste(candidats_hr, rr_hz, hr_prev_hz,
                       snr_min=3.0,
                       tol_harm=0.06,
                       n_harm=5,
                       delta_f=0.35):
    """
    Stratégie :
    1. supprimer les candidats trop faibles
    2. supprimer les candidats proches des harmoniques RR
    3. si HR précédent existe, garder les candidats cohérents temporellement
    4. choisir le meilleur restant
    """

    debug = {
        "candidats_init": [],
        "candidats_valides": [],
        "candidats_temporels": [],
        "raison": "aucun_candidat",
        "harm_rejetee": []
    }

    if len(candidats_hr) == 0:
        return np.nan, debug

    debug["candidats_init"] = [c["freq"] for c in candidats_hr]

    # 1) Seuil SNR
    candidats_snr = [c for c in candidats_hr if c["snr"] >= snr_min]
    if len(candidats_snr) == 0:
        debug["raison"] = "snr_trop_faible"
        return np.nan, debug

    # 2) Rejet harmoniques RR
    candidats_valides = []
    for c in candidats_snr:
        rejet, n_h = est_proche_harmonique(c["freq"], rr_hz, n_harm=n_harm, tol=tol_harm)
        if rejet:
            debug["harm_rejetee"].append((c["freq"], n_h))
        else:
            candidats_valides.append(c)

    debug["candidats_valides"] = [c["freq"] for c in candidats_valides]

    if len(candidats_valides) == 0:
        debug["raison"] = "rejete_harmoniques_rr"
        return np.nan, debug

    # 3) Cohérence temporelle
    if np.isfinite(hr_prev_hz):
        candidats_temporels = [
            c for c in candidats_valides if abs(c["freq"] - hr_prev_hz) <= delta_f
        ]
    else:
        candidats_temporels = []

    debug["candidats_temporels"] = [c["freq"] for c in candidats_temporels]

    # 4) Choix final
    if len(candidats_temporels) > 0:
        # parmi ceux cohérents, on prend le plus puissant
        best = max(candidats_temporels, key=lambda c: c["power"])
        debug["raison"] = "choix_coherent_temporel"
        return best["freq"], debug

    # sinon parmi les valides, prendre le plus proche du HR précédent si connu
    if np.isfinite(hr_prev_hz):
        best = min(candidats_valides, key=lambda c: abs(c["freq"] - hr_prev_hz))
        debug["raison"] = "repli_plus_proche_hr_precedent"
        return best["freq"], debug

    # sinon au tout début, on prend le plus puissant des valides
    best = max(candidats_valides, key=lambda c: c["power"])
    debug["raison"] = "initial_plus_puissant_valide"
    return best["freq"], debug


# ============================================================
# 6) Affichage constellation
# ============================================================
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


# ============================================================
# 7) Buffers / paramètres
# ============================================================
fenetre_signal = 30.0
affichage = 1.0
echantillons_min = 200

buffer_t = deque()
buffer_phi = deque()

last_print = time.time()
fs_est = None
beta_fs = 0.05

# lock bin
idx_lock = None
lock_margin = 2
reacq_period = 40
reacq_count = 0

# tracking HR
hr_prev_hz = np.nan
hr_last_valid_hz = np.nan
hr_last_valid_time = None
hold_max_s = 4.0


# ============================================================
# 8) Sélection bin stable
# ============================================================
def choisir_idx_stable(z_bin, idx_precedent, margin=2):
    amp = np.abs(z_bin)

    if idx_precedent is None:
        return int(np.argmax(amp))

    n = len(amp)
    left = max(0, idx_precedent - margin)
    right = min(n, idx_precedent + margin + 1)

    idx_local = left + int(np.argmax(amp[left:right]))
    return idx_local


def selection_bin_verrouillee(z_bin):
    global idx_lock, reacq_count

    if idx_lock is None or reacq_count >= reacq_period:
        idx_lock = int(np.argmax(np.abs(z_bin)))
        reacq_count = 0
    else:
        idx_lock = choisir_idx_stable(z_bin, idx_lock, margin=lock_margin)

    reacq_count += 1

    left = max(0, idx_lock - 1)
    right = min(len(z_bin), idx_lock + 2)
    z_sel = np.mean(z_bin[left:right])

    return z_sel, idx_lock


# ============================================================
# 9) Boucle principale
# ============================================================
try:
    while True:
        code_erreur, resultat, tableau_IQ = uRAD_RP_SDK11.detection()
        if code_erreur != 0:
            closeProgram()

        I_brut = np.asarray(tableau_IQ[0], dtype=np.float64)
        Q_brut = np.asarray(tableau_IQ[1], dtype=np.float64)
        z_bin = I_brut + 1j * Q_brut

        # ----------------------------------------------------
        # Sélection bin verrouillée
        # ----------------------------------------------------
        z_sel, idx_sel = selection_bin_verrouillee(z_bin)
        I_sel = float(np.real(z_sel))
        Q_sel = float(np.imag(z_sel))

        # ----------------------------------------------------
        # Constellation
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Pipeline phase
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Traitement fréquentiel
        # ----------------------------------------------------
        if len(buffer_phi) >= echantillons_min and fs_est is not None:
            phi_wrap = np.array(buffer_phi, dtype=np.float64)
            phi = np.unwrap(phi_wrap)

            # suppression dérive lente / composante DC
            phi_hp = filtre_pass_haut(phi, fs_est, fc=0.05)

            # bandes physiologiques
            sig_rr = passe_bande(phi_hp, 0.10, 0.50, fs_est)
            sig_hr = passe_bande(phi_hp, 0.80, 2.50, fs_est)

            # analyse RR
            rr_hz_dom, rr2, rr_snr, rr_pow, rr_candidats = analyse_spectrale_debug(
                sig_rr, fs_est, 0.10, 0.50
            )
            rr_hz = rr_hz_dom
            rr_rpm = rr_hz * 60.0 if np.isfinite(rr_hz) else np.nan

            # analyse HR
            hr_hz_dom, hr2, hr_snr, hr_pow, hr_candidats = analyse_spectrale_debug(
                sig_hr, fs_est, 0.80, 2.50
            )

            hr_hz_robuste, hr_debug = choisir_hr_robuste(
                hr_candidats,
                rr_hz,
                hr_prev_hz,
                snr_min=3.0,
                tol_harm=0.06,
                n_harm=5,
                delta_f=0.35
            )

            hr_source = "nouveau"

            # stratégie de hold si rien de valide
            if np.isfinite(hr_hz_robuste):
                hr_prev_hz = hr_hz_robuste
                hr_last_valid_hz = hr_hz_robuste
                hr_last_valid_time = t_now
                hr_final_hz = hr_hz_robuste
            else:
                if hr_last_valid_time is not None and (t_now - hr_last_valid_time) <= hold_max_s:
                    hr_final_hz = hr_last_valid_hz
                    hr_source = "hold_last_valid"
                else:
                    hr_final_hz = np.nan
                    hr_source = "nan"

            hr_bpm = hr_final_hz * 60.0 if np.isfinite(hr_final_hz) else np.nan

            # ------------------------------------------------
            # Debug
            # ------------------------------------------------
            if (t_now - last_print) > affichage:
                harm2 = 2 * rr_hz if np.isfinite(rr_hz) else np.nan
                harm3 = 3 * rr_hz if np.isfinite(rr_hz) else np.nan
                harm4 = 4 * rr_hz if np.isfinite(rr_hz) else np.nan
                harm5 = 5 * rr_hz if np.isfinite(rr_hz) else np.nan

                hr_cand_str = ", ".join([f"{c['freq']:.3f}" for c in hr_candidats[:5]]) if hr_candidats else "aucun"
                hr_valid_str = ", ".join([f"{x:.3f}" for x in hr_debug["candidats_valides"]]) if hr_debug["candidats_valides"] else "aucun"
                hr_temp_str = ", ".join([f"{x:.3f}" for x in hr_debug["candidats_temporels"]]) if hr_debug["candidats_temporels"] else "aucun"

                if len(hr_debug["harm_rejetee"]) > 0:
                    harm_rej_str = ", ".join([f"{f0:.3f}(~{n}xRR)" for f0, n in hr_debug["harm_rejetee"]])
                else:
                    harm_rej_str = "aucun"

                print(
                    f"\n---------------- DEBUG ----------------\n"
                    f"bin_idx            : {idx_sel}\n"
                    f"bin_amplitude      : {abs(z_sel):.4f}\n"
                    f"fs_est             : {fs_est:.2f} Hz\n"
                    f"\n"
                    f"RR dominant        : {rr_hz:.3f} Hz  ({rr_rpm:.2f} rpm)\n"
                    f"RR 2nd peak        : {rr2:.3f} Hz\n"
                    f"RR SNR             : {rr_snr:.2f} dB\n"
                    f"\n"
                    f"HR dominant brut   : {hr_hz_dom:.3f} Hz  ({hr_hz_dom*60.0 if np.isfinite(hr_hz_dom) else np.nan:.2f} bpm)\n"
                    f"HR 2nd peak brut   : {hr2:.3f} Hz\n"
                    f"HR SNR brut        : {hr_snr:.2f} dB\n"
                    f"\n"
                    f"HR candidats       : {hr_cand_str}\n"
                    f"HR valides         : {hr_valid_str}\n"
                    f"HR cohérents temp. : {hr_temp_str}\n"
                    f"HR rejetés harm.   : {harm_rej_str}\n"
                    f"raison choix HR    : {hr_debug['raison']}\n"
                    f"source HR sortie   : {hr_source}\n"
                    f"\n"
                    f"HR final           : {hr_final_hz:.3f} Hz  ({hr_bpm:.2f} bpm)\n"
                    f"HR précédent       : {hr_prev_hz:.3f} Hz\n"
                    f"\n"
                    f"Resp harmonics     : {harm2:.3f}Hz {harm3:.3f}Hz {harm4:.3f}Hz {harm5:.3f}Hz\n"
                    f"---------------------------------------"
                )

                last_print = t_now

except KeyboardInterrupt:
    closeProgram()