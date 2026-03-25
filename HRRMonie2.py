"""
Pipeline d'extraction des signes vitaux (RR & HR) avec le radar uRAD 24 GHz
SDK uRAD_RP_SDK11 - Raspberry Pi 5 (Bookworm)

Auteur : basé sur le cours de M. Mouhamadou (3iL / XLIM)
Références :
  [1] Paterniani et al., "Radar-Based Monitoring of Vital Signs: A Tutorial Overview",
      Proceedings of the IEEE, Vol. 111, No. 3, March 2023
  [2] Chaoyan Zhang et al., "A radar vital signs detection method in complex environments",
      Scientific Reports 16, 2026

API SDK uRAD_RP_SDK11 utilisée :
  uRAD_RP_SDK11.turnON()
  uRAD_RP_SDK11.loadConfiguration(mode, f0, BW, Ns, Ntar, Rmax, MTI, Mth,
                                   Alpha, distance_true, velocity_true,
                                   SNR_true, I_true, Q_true, movement_true)
  code_erreur, resultat, tableau_IQ = uRAD_RP_SDK11.detection()
  uRAD_RP_SDK11.turnOFF()

Pipeline :
  1. Acquisition du signal I/Q brut via detection()
  2. Prétraitement I/Q  : retrait DC, correction déséquilibre I/Q
  3. Estimation de la phase unwrappée
  4. Suppression dérive lente (mouvement global)
  5. Filtrage passe-bande RR puis HR
  6. Estimation des fréquences par FFT glissante
  7. Affichage temps-réel
"""

import time
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.fft import rfft, rfftfreq
import collections

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 : IMPORT SDK
# ──────────────────────────────────────────────────────────────────────────────

SIMULATION_MODE = False

try:
    import uRAD_RP_SDK11
    print("[INFO] SDK uRAD_RP_SDK11 détecté.")
except ImportError:
    print("[WARN] uRAD_RP_SDK11 non trouvé → mode SIMULATION activé.")
    SIMULATION_MODE = True


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 : PARAMÈTRES GLOBAUX
# ──────────────────────────────────────────────────────────────────────────────

class RadarConfig:
    """
    Paramètres de configuration du radar et du traitement du signal.

    ── Paramètres loadConfiguration ──────────────────────────────────────────
    mode     : 1 = CW Doppler (pas de distance, I/Q brut)
               2 = FMCW sawtooth | 3 = FMCW triangular
    f0       : fréquence centrale (GHz) — 24 pour uRAD 24 GHz
    BW       : bande passante FMCW (MHz) — ignoré en mode CW (mettre 0)
    Ns       : nombre d'échantillons par trame (puissance de 2 recommandée)
    Ntar     : nombre de cibles max à détecter (1 pour signes vitaux)
    Rmax     : portée max (m)
    MTI      : 0 = désactivé | 1 = Moving Target Indicator (supprime statique)
    Mth      : seuil de détection MTI (0 si MTI=0)
    Alpha    : facteur de lissage MTI (0 si MTI=0)
    distance_true  : True → retourne les distances dans resultat
    velocity_true  : True → retourne les vitesses
    SNR_true       : True → retourne le SNR
    I_true         : True → retourne les échantillons I dans tableau_IQ
    Q_true         : True → retourne les échantillons Q dans tableau_IQ
    movement_true  : True → retourne un indicateur de mouvement
    """

    # ── Paramètres SDK loadConfiguration ──────────────────────────────────
    mode           = 1       # CW Doppler → tableau_IQ contient I et Q bruts
    f0             = 24      # GHz
    BW             = 0       # MHz (non utilisé en CW)
    Ns             = 200     # Échantillons par appel detection()
                             # → fs_effective ≈ Ns / T_trame
                             # En CW le SDK retourne Ns paires I/Q par trame
    Ntar           = 1
    Rmax           = 5       # m
    MTI            = 0       # Pas de MTI en CW (pas de dimension distance)
    Mth            = 0
    Alpha          = 0
    distance_true  = False
    velocity_true  = False
    SNR_true       = False
    I_true         = True    # ← obligatoire pour récupérer le signal I/Q
    Q_true         = True    # ← obligatoire
    movement_true  = False

    # ── Fréquence d'échantillonnage effective ──────────────────────────────
    # En mode CW le radar renvoie `Ns` paires I/Q par appel detection().
    # On appelle detection() aussi vite que possible → fs ≈ Ns / T_appel.
    # Pour contrôler fs on peut fixer Ns petit (ex: 1) et cadencer la boucle.
    # Ici on fixe Ns=200 et on consomme les Ns échantillons d'un coup.
    FS_HZ          = 200.0   # Hz — doit correspondre au taux réel du radar

    LAMBDA_M       = 3e8 / (f0 * 1e9)   # λ ≈ 0.01249 m

    # ── Durée d'acquisition ────────────────────────────────────────────────
    WINDOW_S       = 30      # Fenêtre FFT (s) → résolution spectrale = 1/30 Hz
    OVERLAP_S      = 5       # Pas de glissement (s)
    BUFFER_S       = 60      # Buffer circulaire max (s)

    # ── Bandes passantes physiologiques ───────────────────────────────────
    RR_LOW_HZ      = 0.1     # Respiration : 6–30 cycles/min
    RR_HIGH_HZ     = 0.5
    HR_LOW_HZ      = 0.8     # Cœur : 48–150 bpm
    HR_HIGH_HZ     = 2.5

    # ── Filtres ───────────────────────────────────────────────────────────
    FILTER_ORDER   = 4
    HP_CUTOFF_HZ   = 0.05    # Passe-haut : supprime DC + mouvement < 3 cpm

    # ── Seuil qualité pic FFT ─────────────────────────────────────────────
    MIN_PEAK_SNR_DB = 3.0


CFG = RadarConfig()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 : SIMULATION (debug sans matériel)
# ──────────────────────────────────────────────────────────────────────────────

class SimulatedRadar:
    """
    Génère un tableau_IQ synthétique de longueur Ns reproduisant :
      respiration + cœur + 2e harmonique respiratoire + bruit gaussien.

    Modèle (cf. [1] eq. 6–8 et [2] eq. 3–5) :
        x(t) = Ar·cos(2π·fr·t) + Ah·cos(2π·fh·t)
        φ(t) = -(4π/λ)·(R0 + x(t))
        z(t) = exp(jφ(t)) + bruit
    """

    def __init__(self):
        self.t = 0.0
        self.dt = 1.0 / CFG.FS_HZ
        self.fr = 0.25          # RR simulé = 15 cpm
        self.fh = 1.10          # HR simulé = 66 bpm
        self.Ar = 2e-3          # m
        self.Ah = 0.5e-3        # m
        self.R0 = 1.0           # m

    def detection(self):
        """
        Imite uRAD_RP_SDK11.detection().
        Retourne (code_erreur=0, resultat=None, tableau_IQ)
        où tableau_IQ = [I_array, Q_array], chacun de longueur Ns.
        """
        Ns = CFG.Ns
        t_vec = self.t + np.arange(Ns) * self.dt
        xr  = self.Ar * np.cos(2 * np.pi * self.fr  * t_vec)
        xh  = self.Ah * np.cos(2 * np.pi * self.fh  * t_vec)
        xr2 = 0.3 * self.Ar * np.cos(2 * np.pi * 2 * self.fr * t_vec)
        x   = xr + xh + xr2
        phi = -(4 * np.pi / CFG.LAMBDA_M) * (self.R0 + x)
        noise_i = np.random.randn(Ns) * 1e-4
        noise_q = np.random.randn(Ns) * 1e-4
        I_arr = np.cos(phi) + noise_i
        Q_arr = np.sin(phi) + noise_q
        self.t += Ns * self.dt
        return 0, None, [I_arr.tolist(), Q_arr.tolist()]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 : INTERFACE RADAR — wrappeur autour de uRAD_RP_SDK11
# ──────────────────────────────────────────────────────────────────────────────

def _close_program():
    """Éteint le radar et lève SystemExit (conforme au manuel SDK)."""
    if not SIMULATION_MODE:
        uRAD_RP_SDK11.turnOFF()
    raise SystemExit


class RadarInterface:
    """
    Encapsule l'initialisation et la lecture uRAD_RP_SDK11.

    Cycle de vie :
        radar = RadarInterface()   # turnON + loadConfiguration
        z_vec = radar.read_frame() # detection() → vecteur complexe de longueur Ns
        radar.close()              # turnOFF
    """

    def __init__(self):
        self._sim = None

        if SIMULATION_MODE:
            self._sim = SimulatedRadar()
            print("[INFO] Radar simulé initialisé (Ns={}).".format(CFG.Ns))
            return

        # ── turnON ────────────────────────────────────────────────────────
        ret = uRAD_RP_SDK11.turnON()
        if ret != 0:
            print(f"[ERREUR] turnON() → code {ret}")
            _close_program()

        # ── loadConfiguration ─────────────────────────────────────────────
        ret = uRAD_RP_SDK11.loadConfiguration(
            CFG.mode,
            CFG.f0,
            CFG.BW,
            CFG.Ns,
            CFG.Ntar,
            CFG.Rmax,
            CFG.MTI,
            CFG.Mth,
            CFG.Alpha,
            CFG.distance_true,
            CFG.velocity_true,
            CFG.SNR_true,
            CFG.I_true,
            CFG.Q_true,
            CFG.movement_true,
        )
        if ret != 0:
            print(f"[ERREUR] loadConfiguration() → code {ret}")
            _close_program()

        print("[INFO] Radar uRAD physique initialisé (mode={}, f0={}GHz, Ns={}).".format(
            CFG.mode, CFG.f0, CFG.Ns))

    def read_frame(self) -> np.ndarray:
        """
        Appelle detection() et retourne un vecteur complexe numpy de taille Ns.

        tableau_IQ = [I_list, Q_list]  (longueur Ns chacun)
        → z = I + jQ
        """
        if self._sim is not None:
            code_erreur, _, tableau_IQ = self._sim.detection()
        else:
            code_erreur, _, tableau_IQ = uRAD_RP_SDK11.detection()

        if code_erreur != 0:
            print(f"[ERREUR] detection() → code {code_erreur}")
            _close_program()

        I_arr = np.asarray(tableau_IQ[0], dtype=float)
        Q_arr = np.asarray(tableau_IQ[1], dtype=float)
        return I_arr + 1j * Q_arr

    def close(self):
        """Éteint le radar proprement."""
        if not SIMULATION_MODE:
            uRAD_RP_SDK11.turnOFF()
            print("[INFO] Radar éteint.")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 : PRÉ-TRAITEMENT I/Q
# ──────────────────────────────────────────────────────────────────────────────

class IQPreprocessor:
    """
    Traitement frame par frame (vecteur de Ns échantillons).

    Étapes :
      1. Soustraction du DC (moyenne sur un historique glissant long)
      2. Correction du déséquilibre I/Q (égalisation des gains I et Q)
         → méthode simple : ratio des écarts-types sur l'historique
    """

    def __init__(self, dc_history_frames: int = 10):
        """
        dc_history_frames : nombre de trames conservées pour estimer le DC.
        """
        self.dc_history_frames = dc_history_frames
        # Stocke les Ns*dc_history_frames derniers échantillons I et Q
        maxlen = CFG.Ns * dc_history_frames
        self.i_hist = collections.deque(maxlen=maxlen)
        self.q_hist = collections.deque(maxlen=maxlen)

    def process(self, z_frame: np.ndarray) -> np.ndarray:
        """
        z_frame : vecteur complexe numpy de longueur Ns (I + jQ).
        Retourne un vecteur complexe centré et équilibré.
        """
        I = z_frame.real.copy()
        Q = z_frame.imag.copy()

        # Mise à jour de l'historique
        self.i_hist.extend(I.tolist())
        self.q_hist.extend(Q.tolist())

        # ── Retrait DC ────────────────────────────────────────────────────
        I_dc = np.mean(self.i_hist)
        Q_dc = np.mean(self.q_hist)
        I -= I_dc
        Q -= Q_dc

        # ── Correction déséquilibre I/Q ───────────────────────────────────
        std_i = np.std(self.i_hist)
        std_q = np.std(self.q_hist)
        if std_i > 1e-12 and std_q > 1e-12:
            Q = Q * (std_i / std_q)

        return I + 1j * Q


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 : EXTRACTION DE PHASE
# ──────────────────────────────────────────────────────────────────────────────

def extract_unwrapped_phase(z_buffer: np.ndarray) -> np.ndarray:
    """
    Calcule la phase unwrappée à partir du buffer de signaux complexes.

    φ[n] = angle(z[n])  →  unwrap  →  φ_unwrapped[n]

    Le unwrapping corrige les sauts de ±π (cf. [1] Section V-A).
    """
    phase_wrapped = np.angle(z_buffer)
    phase_unwrapped = np.unwrap(phase_wrapped)
    return phase_unwrapped


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 : FILTRES EN TEMPS RÉEL (SOS + état)
# ──────────────────────────────────────────────────────────────────────────────

def make_bandpass_sos(low_hz, high_hz, fs, order=4):
    """Butterworth passe-bande via représentation Second-Order Sections."""
    nyq = fs / 2.0
    low  = max(low_hz  / nyq, 1e-4)
    high = min(high_hz / nyq, 1 - 1e-4)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def make_highpass_sos(cutoff_hz, fs, order=2):
    """Butterworth passe-haut pour supprimer la dérive lente."""
    nyq = fs / 2.0
    cut = max(cutoff_hz / nyq, 1e-4)
    sos = butter(order, cut, btype='high', output='sos')
    return sos


class RealtimeFilter:
    """
    Filtre en temps réel avec maintien de l'état entre les appels,
    ce qui évite les discontinuités dans le signal filtré.
    """

    def __init__(self, sos: np.ndarray):
        self.sos = sos
        self.zi  = sosfilt_zi(sos)  # état initial = zéro
        self._initialized = False

    def filter_sample(self, x: float) -> float:
        y, self.zi = sosfilt(self.sos, [x], zi=self.zi)
        return float(y[0])

    def filter_array(self, x: np.ndarray) -> np.ndarray:
        """Filtre un vecteur en maintenant l'état."""
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 : ESTIMATEUR DE FRÉQUENCE (FFT sur fenêtre glissante)
# ──────────────────────────────────────────────────────────────────────────────

def estimate_frequency_fft(signal: np.ndarray, fs: float,
                            f_low: float, f_high: float,
                            zero_pad_factor: int = 4) -> tuple[float, float]:
    """
    Estime la fréquence dominante dans [f_low, f_high] par FFT.

    Paramètres
    ----------
    signal         : tableau 1D du signal (phase ou signal filtré)
    fs             : fréquence d'échantillonnage (Hz)
    f_low, f_high  : bornes de la bande d'intérêt (Hz)
    zero_pad_factor: facteur de sur-échantillonnage spectral

    Retourne
    --------
    f_est   : fréquence estimée (Hz), -1 si non fiable
    snr_db  : SNR du pic (dB)
    """
    N = len(signal)
    if N < 2:
        return -1.0, 0.0

    # Fenêtrage de Hann pour réduire la fuite spectrale
    window = np.hanning(N)
    sig_w  = signal * window

    # FFT avec zero-padding
    N_fft  = N * zero_pad_factor
    freqs  = rfftfreq(N_fft, d=1.0 / fs)
    spectrum = np.abs(rfft(sig_w, n=N_fft))

    # Restriction à la bande d'intérêt
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return -1.0, 0.0

    band_spectrum = spectrum[mask]
    band_freqs    = freqs[mask]

    # Fréquence du pic
    peak_idx = np.argmax(band_spectrum)
    f_est    = band_freqs[peak_idx]
    peak_amp = band_spectrum[peak_idx]

    # SNR simple : pic vs médiane du spectre de bande
    noise_floor = np.median(band_spectrum)
    snr_db = 20 * np.log10(peak_amp / (noise_floor + 1e-12))

    return f_est, snr_db


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 : PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

class VitalSignsPipeline:
    """
    Orchestre l'ensemble du traitement :
      radar → IQ preproc → phase → filtrage → FFT → RR / HR

    Chaque appel à radar.read_frame() retourne Ns échantillons d'un coup.
    Le pipeline pousse ces Ns échantillons dans les buffers circulaires
    puis déclenche une estimation toutes les OVERLAP_S secondes.
    """

    def __init__(self):
        self.radar    = RadarInterface()
        self.iq_prep  = IQPreprocessor(dc_history_frames=10)

        # ── Filtres temps-réel SOS ────────────────────────────────────────
        sos_hp = make_highpass_sos(CFG.HP_CUTOFF_HZ,  CFG.FS_HZ)
        sos_rr = make_bandpass_sos(CFG.RR_LOW_HZ,  CFG.RR_HIGH_HZ, CFG.FS_HZ, CFG.FILTER_ORDER)
        sos_hr = make_bandpass_sos(CFG.HR_LOW_HZ,  CFG.HR_HIGH_HZ, CFG.FS_HZ, CFG.FILTER_ORDER)
        self.filt_hp  = RealtimeFilter(sos_hp)
        self.filt_rr  = RealtimeFilter(sos_rr)
        self.filt_hr  = RealtimeFilter(sos_hr)

        # ── Buffers circulaires ───────────────────────────────────────────
        buf_size = int(CFG.BUFFER_S * CFG.FS_HZ)
        self.ph_buf  = collections.deque(maxlen=buf_size)  # phase brute
        self.rr_buf  = collections.deque(maxlen=buf_size)  # signal filtré RR
        self.hr_buf  = collections.deque(maxlen=buf_size)  # signal filtré HR

        # ── Historique des estimations ────────────────────────────────────
        self.rr_history = []
        self.hr_history = []
        self.ts_history = []

        # ── Compteurs ─────────────────────────────────────────────────────
        self.win_samples  = int(CFG.WINDOW_S  * CFG.FS_HZ)
        self.step_samples = int(CFG.OVERLAP_S * CFG.FS_HZ)
        self.total_samples = 0
        self.last_est_sample = -self.step_samples  # force estimation dès que possible

    # ── 9.1 Traitement d'une frame (Ns échantillons) ──────────────────────

    def _process_frame(self, z_frame: np.ndarray):
        """
        z_frame : vecteur complexe numpy de longueur Ns issu de read_frame().
        Pousse les échantillons dans les buffers après pré-traitement.
        """
        # Pré-traitement I/Q (frame entière)
        z_clean = self.iq_prep.process(z_frame)

        # Phase instantanée échantillon par échantillon (angle + unwrap différé)
        phi_arr = np.angle(z_clean)

        # Filtrage passe-haut sur le vecteur
        phi_hp = self.filt_hp.filter_array(phi_arr)

        # Filtrage passe-bande RR et HR
        sig_rr = self.filt_rr.filter_array(phi_hp)
        sig_hr = self.filt_hr.filter_array(phi_hp)

        # Stockage dans les buffers
        self.ph_buf.extend(phi_arr.tolist())
        self.rr_buf.extend(sig_rr.tolist())
        self.hr_buf.extend(sig_hr.tolist())

        self.total_samples += len(z_frame)

    # ── 9.2 Estimation sur la fenêtre courante ─────────────────────────────

    def _estimate_on_window(self) -> dict:
        """
        Calcule RR et HR sur les win_samples derniers échantillons.
        Utilise la phase unwrappée pour une meilleure résolution spectrale.
        """
        n = min(len(self.ph_buf), self.win_samples)
        if n < self.win_samples // 2:
            return {}

        ph_arr = np.array(list(self.ph_buf)[-n:])
        rr_arr = np.array(list(self.rr_buf)[-n:])
        hr_arr = np.array(list(self.hr_buf)[-n:])

        # Unwrap + détrend sur la fenêtre
        ph_uw = np.unwrap(ph_arr)
        ph_uw -= np.polyval(np.polyfit(np.arange(n), ph_uw, 1), np.arange(n))

        rr_hz, rr_snr = estimate_frequency_fft(rr_arr, CFG.FS_HZ,
                                                CFG.RR_LOW_HZ, CFG.RR_HIGH_HZ)
        hr_hz, hr_snr = estimate_frequency_fft(hr_arr, CFG.FS_HZ,
                                                CFG.HR_LOW_HZ, CFG.HR_HIGH_HZ)

        result = {"timestamp_s": time.time(), "n_samples": n}

        result["RR_bpm"]    = rr_hz * 60.0 if (rr_snr >= CFG.MIN_PEAK_SNR_DB and rr_hz > 0) else None
        result["RR_snr_db"] = rr_snr
        result["HR_bpm"]    = hr_hz * 60.0 if (hr_snr >= CFG.MIN_PEAK_SNR_DB and hr_hz > 0) else None
        result["HR_snr_db"] = hr_snr
        return result

    # ── 9.3 Boucle principale ─────────────────────────────────────────────

    def run(self, duration_s: float = 120.0, verbose: bool = True):
        """
        Lance l'acquisition et le traitement pendant `duration_s` secondes.
        Chaque itération lit une frame de Ns échantillons via detection().
        Une estimation est produite toutes les OVERLAP_S secondes.
        """
        print(f"\n{'='*60}")
        print(f"  Pipeline Signes Vitaux — uRAD {CFG.f0} GHz")
        print(f"  Durée : {duration_s:.0f}s | Fenêtre : {CFG.WINDOW_S}s")
        print(f"  Ns={CFG.Ns} | fs={CFG.FS_HZ} Hz | Mode={'SIM' if SIMULATION_MODE else 'HW'}")
        print(f"{'='*60}\n")

        t_start = time.time()

        try:
            while True:
                t_now = time.time()
                if t_now - t_start >= duration_s:
                    break

                # ── Lecture trame radar ───────────────────────────────────
                z_frame = self.radar.read_frame()   # appel detection()
                self._process_frame(z_frame)

                # ── Estimation si assez de nouveaux échantillons ──────────
                samples_since_last = self.total_samples - self.last_est_sample
                if samples_since_last >= self.step_samples:
                    result = self._estimate_on_window()
                    if result:
                        self.last_est_sample = self.total_samples
                        self._store_result(result)
                        if verbose:
                            elapsed = t_now - t_start
                            self._print_result(result, elapsed)

        except KeyboardInterrupt:
            print("\n[INFO] Arrêt manuel.")
        finally:
            self.radar.close()
            print("\n[INFO] Pipeline terminé.")
            self._print_summary()

    def _store_result(self, result: dict):
        self.ts_history.append(result.get("timestamp_s", time.time()))
        self.rr_history.append(result.get("RR_bpm"))
        self.hr_history.append(result.get("HR_bpm"))

    def _print_result(self, result: dict, elapsed_s: float):
        rr     = result.get("RR_bpm")
        hr     = result.get("HR_bpm")
        rr_snr = result.get("RR_snr_db", 0.0)
        hr_snr = result.get("HR_snr_db", 0.0)
        rr_str = f"{rr:5.1f} bpm (SNR={rr_snr:4.1f} dB)" if rr else "  N/A  (SNR trop faible)"
        hr_str = f"{hr:5.1f} bpm (SNR={hr_snr:4.1f} dB)" if hr else "  N/A  (SNR trop faible)"
        print(f"[{elapsed_s:6.1f}s]  RR : {rr_str}   |   HR : {hr_str}")

    def _print_summary(self):
        valid_rr = [v for v in self.rr_history if v is not None]
        valid_hr = [v for v in self.hr_history if v is not None]
        print(f"\n{'─'*60}")
        print(f"  RÉSUMÉ  ({len(self.rr_history)} estimations)")
        print(f"{'─'*60}")
        if valid_rr:
            print(f"  RR moyen : {np.mean(valid_rr):.1f} bpm  ±{np.std(valid_rr):.1f}  "
                  f"[{np.min(valid_rr):.1f} – {np.max(valid_rr):.1f}]")
        else:
            print("  RR : aucune estimation valide")
        if valid_hr:
            print(f"  HR moyen : {np.mean(valid_hr):.1f} bpm  ±{np.std(valid_hr):.1f}  "
                  f"[{np.min(valid_hr):.1f} – {np.max(valid_hr):.1f}]")
        else:
            print("  HR : aucune estimation valide")
        print(f"{'─'*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 : UTILITAIRE — SAUVEGARDE CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_results_csv(pipeline: VitalSignsPipeline, path: str = "vital_signs.csv"):
    """Exporte l'historique des estimations dans un fichier CSV."""
    import csv
    t0 = pipeline.ts_history[0] if pipeline.ts_history else 0.0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_s", "RR_bpm", "HR_bpm"])
        for ts, rr, hr in zip(pipeline.ts_history,
                               pipeline.rr_history,
                               pipeline.hr_history):
            writer.writerow([
                f"{ts - t0:.2f}",
                f"{rr:.1f}" if rr is not None else "",
                f"{hr:.1f}" if hr is not None else "",
            ])
    print(f"[INFO] Résultats sauvegardés → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11 : LANCEMENT DIRECT
# ──────────────────────────────────────────────────────────────────────────────
# Utilisation :
#   python vital_signs_urad.py
#   python vital_signs_urad.py --duration 60 --csv out.csv --sim

import argparse

_parser = argparse.ArgumentParser(
    description="Extraction des signes vitaux avec radar uRAD 24 GHz")
_parser.add_argument("--duration", type=float, default=120.0,
                     help="Durée d'acquisition en secondes (défaut: 120)")
_parser.add_argument("--window",   type=float, default=CFG.WINDOW_S,
                     help="Durée de la fenêtre FFT (défaut: 30 s)")
_parser.add_argument("--overlap",  type=float, default=CFG.OVERLAP_S,
                     help="Pas de glissement (défaut: 5 s)")
_parser.add_argument("--csv",      type=str,   default="",
                     help="Chemin du fichier CSV de sortie (optionnel)")
_parser.add_argument("--sim",      action="store_true",
                     help="Force le mode simulation (sans radar physique)")
_args = _parser.parse_args()

if _args.sim:
    SIMULATION_MODE = True

CFG.WINDOW_S  = _args.window
CFG.OVERLAP_S = _args.overlap

pipeline = VitalSignsPipeline()
pipeline.run(duration_s=_args.duration, verbose=True)

if _args.csv:
    save_results_csv(pipeline, _args.csv)

