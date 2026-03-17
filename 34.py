import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, detrend
import uRAD_RP_SDK11


# =========================
# PARAMETRES
# =========================
WINDOW_SECONDS = 30.0          # fenêtre d'analyse
MIN_SECONDS = 12.0             # minimum avant 1re estimation
PRINT_EVERY = 1.0              # affichage toutes les 1 s

# Bandes vitales usuelles
RESP_BAND = (0.10, 0.50)       # Hz  -> 6 à 30 rpm
HEART_BAND = (0.80, 2.00)      # Hz  -> 48 à 120 bpm

# Configuration uRAD
# Mode 1 = CW : plus simple pour suivre la phase I/Q dans le temps.
mode = 1
f0 = 120                       # entier, plage CW 5..245
BW = 50                        # ignoré en mode 1, mais argument obligatoire
Ns = 200                       # 50..200
Ntar = 1
Rmax = 75                      # en mode 1, sert côté firmware mais n'est pas critique ici
MTI = 0
Mth = 4
Alpha = 20

distance_true = False
velocity_true = False
SNR_true = False
I_true = True
Q_true = True
movement_true = False


# =========================
# OUTILS SIGNAL
# =========================
def bandpass(x, fs, f_lo, f_hi, order=4):
    nyq = 0.5 * fs
    lo = f_lo / nyq
    hi = f_hi / nyq
    if hi >= 1.0:
        hi = 0.999
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, x)


def dominant_freq_hz(x, fs, f_lo, f_hi):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    n = len(x)
    if n < 8:
        return None

    win = np.hanning(n)
    spec = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(spec)

    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return None

    f_sel = freqs[mask]
    m_sel = mag[mask]
    idx = np.argmax(m_sel)
    return float(f_sel[idx])


def rpm_from_hz(f):
    return None if f is None else 60.0 * f


def bpm_from_hz(f):
    return None if f is None else 60.0 * f


# =========================
# PIPELINE VITAL SIGNS
# =========================
def estimate_block_phase(I_raw, Q_raw):
    """
    Une acquisition uRAD renvoie un bloc I/Q.
    Pour obtenir une série lente exploitable pour RR/HR,
    on résume chaque bloc par une phase complexe moyenne.
    """
    I = np.asarray(I_raw, dtype=float)
    Q = np.asarray(Q_raw, dtype=float)

    # retrait DC / offset IQ
    I = I - np.mean(I)
    Q = Q - np.mean(Q)

    z = I + 1j * Q
    z_mean = np.mean(z)

    # évite les blocs trop faibles
    if np.abs(z_mean) < 1e-12:
        return None

    return np.angle(z_mean)


def analyze_phase_series(t_slow, phi_slow):
    """
    t_slow  : timestamps des blocs
    phi_slow: phase moyenne par bloc
    """
    t_slow = np.asarray(t_slow, dtype=float)
    phi_slow = np.asarray(phi_slow, dtype=float)

    if len(phi_slow) < 32:
        return None, None, None

    # phase continue
    phi = np.unwrap(phi_slow)

    # estimation fs lente depuis les timestamps
    dt = np.diff(t_slow)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None, None, None

    fs_slow = 1.0 / np.median(dt)

    # retrait dérive lente
    phi = detrend(phi)

    # filtrages
    try:
        resp_sig = bandpass(phi, fs_slow, RESP_BAND[0], RESP_BAND[1], order=4)
        heart_sig = bandpass(phi, fs_slow, HEART_BAND[0], HEART_BAND[1], order=4)
    except Exception:
        return None, None, fs_slow

    f_resp = dominant_freq_hz(resp_sig, fs_slow, RESP_BAND[0], RESP_BAND[1])
    f_heart = dominant_freq_hz(heart_sig, fs_slow, HEART_BAND[0], HEART_BAND[1])

    rr_rpm = rpm_from_hz(f_resp)
    hr_bpm = bpm_from_hz(f_heart)

    return rr_rpm, hr_bpm, fs_slow


# =========================
# MAIN
# =========================
def main():
    rc = uRAD_RP_SDK11.turnON()
    if rc != 0:
        raise RuntimeError(f"turnON a échoué: code={rc}")

    try:
        rc = uRAD_RP_SDK11.loadConfiguration(
            mode, f0, BW, Ns, Ntar, Rmax, MTI, Mth, Alpha,
            distance_true, velocity_true, SNR_true,
            I_true, Q_true, movement_true
        )
        if rc != 0:
            raise RuntimeError(f"loadConfiguration a échoué: code={rc}")

        t_buffer = deque()
        phi_buffer = deque()
        last_print = 0.0

        while True:
            t0 = time.time()

            rc, results, raw_results = uRAD_RP_SDK11.detection()
            if rc != 0:
                print(f"[WARN] detection() code={rc}")
                continue

            # SDK: raw_results contient les données brutes demandées.
            # Ici on a activé I_true et Q_true.
            I_raw = raw_results[0]
            Q_raw = raw_results[1]

            phi = estimate_block_phase(I_raw, Q_raw)
            if phi is None:
                continue

            now = time.time()
            t_buffer.append(now)
            phi_buffer.append(phi)

            # conserve seulement WINDOW_SECONDS
            while len(t_buffer) > 1 and (now - t_buffer[0]) > WINDOW_SECONDS:
                t_buffer.popleft()
                phi_buffer.popleft()

            # attendre un minimum de données
            if (t_buffer[-1] - t_buffer[0]) < MIN_SECONDS:
                continue

            if (now - last_print) >= PRINT_EVERY:
                rr_rpm, hr_bpm, fs_slow = analyze_phase_series(list(t_buffer), list(phi_buffer))

                if rr_rpm is not None and hr_bpm is not None:
                    print(f"RR = {rr_rpm:5.1f} resp/min | HR = {hr_bpm:5.1f} bpm | fs_slow = {fs_slow:4.1f} Hz")
                else:
                    print("Analyse en cours...")

                last_print = now

            # petite détente CPU
            elapsed = time.time() - t0
            if elapsed < 0.005:
                time.sleep(0.005 - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            uRAD_RP_SDK11.turnOFF()
        except Exception:
            pass


if __name__ == "__main__":
    main()