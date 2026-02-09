import time
import math
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import sosfiltfilt, find_peaks
import uRAD_RP_SDK11


# =========================
# Configuration radar uRAD
# =========================
mode = 3          # 4 ?
f0 = 5
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
# Paramètres acquisition
# =========================
t = 0.05
fs = 1.0 / t

phiUnwrappedBuffer = []


def closeProgram():
    uRAD_RP_SDK11.turnOFF()
    raise SystemExit


# =========================
# Init radar
# =========================
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
# Traitement I/Q -> complexe
# =========================
def IQRawToComplex(I_12, Q_12, W=200, reset=False):
    """
    Prend une frame I/Q (arrays) et retourne un seul échantillon complexe "utile"
    (bin le plus énergétique) + I/Q corrigés DC + index du bin.
    """
    I_12 = np.asarray(I_12, dtype=np.float64)
    Q_12 = np.asarray(Q_12, dtype=np.float64)

    # Bin le plus énergétique
    power = I_12**2 + Q_12**2
    idx = int(np.argmax(power))
    I_raw = I_12[idx]
    Q_raw = Q_12[idx]

    # Init / reset buffers moyenne glissante
    if reset or not hasattr(IQRawToComplex, "bufI") or IQRawToComplex.bufI.size != W:
        IQRawToComplex.bufI = np.zeros(W, dtype=np.float64)
        IQRawToComplex.bufQ = np.zeros(W, dtype=np.float64)
        IQRawToComplex.sumI = 0.0
        IQRawToComplex.sumQ = 0.0
        IQRawToComplex.i = 0
        IQRawToComplex.n = 0

    # Recentrage 12 bits (0..4095 -> centré)
    I0 = I_raw - 2047.5
    Q0 = Q_raw - 2047.5

    # Mise à jour moyenne glissante
    j = IQRawToComplex.i
    IQRawToComplex.sumI += I0 - IQRawToComplex.bufI[j]
    IQRawToComplex.sumQ += Q0 - IQRawToComplex.bufQ[j]
    IQRawToComplex.bufI[j] = I0
    IQRawToComplex.bufQ[j] = Q0

    IQRawToComplex.i = (j + 1) % W
    IQRawToComplex.n = min(IQRawToComplex.n + 1, W)

    I_dc = IQRawToComplex.sumI / IQRawToComplex.n
    Q_dc = IQRawToComplex.sumQ / IQRawToComplex.n

    I_corr = I0 - I_dc
    Q_corr = Q0 - Q_dc

    return (I_corr + 1j * Q_corr), I_corr, Q_corr, idx

# =========================
# Unwrap incrémental (phase continue)
# =========================
def unwrapIncremental(phi, reset=False):
    if reset or not hasattr(unwrapIncremental, "prev"):
        unwrapIncremental.prev = phi
        unwrapIncremental.offset = 0.0
        return phi

    deltaphi = phi - unwrapIncremental.prev

    if deltaphi > np.pi:
        unwrapIncremental.offset -= 2.0 * np.pi
    elif deltaphi < -np.pi:
        unwrapIncremental.offset += 2.0 * np.pi

    unwrapIncremental.prev = phi
    return phi + unwrapIncremental.offset
def bandpass_sos(x, fs, f1, f2, order=4):
    sos = butter(order, [f1/(fs/2), f2/(fs/2)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)

def fft_peak(x, fs, fmin, fmax):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), d=1.0/fs)
    P = (np.abs(X) ** 2)

    m = (f >= fmin) & (f <= fmax)
    f2, P2 = f[m], P[m]
    if len(P2) < 5:
        return None

    peaks, props = find_peaks(P2, prominence=np.max(P2)*0.02)
    if len(peaks) == 0:
        k = int(np.argmax(P2))
        peak_f = float(f2[k])
        peak_p = float(P2[k])
        prom = 0.0
    else:
        i = int(np.argmax(P2[peaks]))
        k = int(peaks[i])
        peak_f = float(f2[k])
        peak_p = float(P2[k])
        prom = float(props["prominences"][i])

    noise = float(np.median(P2))
    snr = peak_p / max(1e-12, noise)
    return peak_f, snr, prom

# =========================
# Filtre passe-haut (calculé 1 fois)
# =========================
# cutoff = 0.05 Hz (comme ton code)
b, a = butter(2, 0.05 / (fs / 2.0), btype="high")
minLen = max(3 * max(len(a), len(b)), 30)


# =========================
# Boucle principale
# =========================
try:
    while True:
        return_code, results, raw_results = uRAD_RP_SDK11.detection()
        if return_code != 0:
            print("Erreur radar")
            break

        I = raw_results[0]
        Q = raw_results[1]

        sig, I_corr, Q_corr, idx = IQRawToComplex(I, Q)

        # Phase + unwrap incrémental
        phi = np.arctan2(Q_corr, I_corr)
        phiUnwrapped = unwrapIncremental(phi)
        phiUnwrappedBuffer.append(phiUnwrapped)

        print(f"I={I_corr:+8.3f} | Q={Q_corr:+8.3f} | Phase={phiUnwrapped:+8.3f} | bin={idx}")

        # Nettoyage par filtfilt quand buffer assez long
        if len(phiUnwrappedBuffer) >= max(50, minLen):
            phiClean = filtfilt(b, a, np.asarray(phiUnwrappedBuffer, dtype=np.float64))
            print("phiClean", float(phiClean[-1]))
        # ===== Suppression dérive & artefacts lents =====

        A = math.sqrt(I_corr*I_corr + Q_corr*Q_corr)

        c = 299_792_458.0
        fc = 24.005e9
        lam = c / fc
        x = (lam / (4.0 * math.pi)) * phiUnwrapped

        if not hasattr(closeProgram, "init_drift"):
            closeProgram.init_drift = True

            tau_base = 10.0
            closeProgram.alpha0 = 1.0 / max(1.0, tau_base * fs)
            closeProgram.base = x

            closeProgram.A_mu = A
            closeProgram.A_var = 0.0

            closeProgram.Wvar = int(max(10, round(2.0 * fs)))
            closeProgram.xhp_buf = np.zeros(closeProgram.Wvar)
            closeProgram.k = 0
            closeProgram.nv = 0

        betaA = 1.0 / max(1.0, 5.0 * fs)
        dA = A - closeProgram.A_mu
        closeProgram.A_mu += betaA * dA
        closeProgram.A_var = (1.0 - betaA) * closeProgram.A_var + betaA * (dA * dA)
        A_sigma = math.sqrt(max(1e-12, closeProgram.A_var))

        A_thr = max(closeProgram.A_mu - 3.0 * A_sigma, 1e-6)
        valid_amp = (A >= A_thr)

        w = 0.0
        if valid_amp:
            w = (A - A_thr) / max(1e-6, closeProgram.A_mu - A_thr)
            w = min(max(w, 0.0), 1.0)

        alpha = closeProgram.alpha0 * w
        closeProgram.base = (1.0 - alpha) * closeProgram.base + alpha * x
        x_hp = x - closeProgram.base

        j = closeProgram.k
        closeProgram.xhp_buf[j] = x_hp
        closeProgram.k = (j + 1) % closeProgram.Wvar
        closeProgram.nv = min(closeProgram.nv + 1, closeProgram.Wvar)

        movement = False
        if closeProgram.nv >= int(fs):
            v = float(np.var(closeProgram.xhp_buf[:closeProgram.nv]))
            movement = (v > 4e-6)

        valid = valid_amp and (not movement)

        print(f"A={A:7.2f} | valid={valid} | x_hp(mm)={x_hp*1e3:+6.3f}")

                # ===== init (une fois) =====
        if not hasattr(closeProgram, "spec_init"):
            closeProgram.spec_init = True
            closeProgram.buf_all = []             # x_hp valides
            closeProgram.t_last_rr = time.time()
            closeProgram.t_last_hr = time.time()

        # ===== à mettre DANS ton while, après x_hp + valid =====
        if valid:
            closeProgram.buf_all.append(float(x_hp))

        # garde ~40s max
        max_keep = int(40 * fs)
        if len(closeProgram.buf_all) > max_keep:
            closeProgram.buf_all = closeProgram.buf_all[-max_keep:]

        now = time.time()

        # ===== Respiration (RR) : 0.1–0.5 Hz, fenêtre 20s, update ~1s =====
        W_rr = int(20 * fs)
        if len(closeProgram.buf_all) >= W_rr and (now - closeProgram.t_last_rr) >= 1.0:
            xw = np.array(closeProgram.buf_all[-W_rr:], dtype=np.float64)
            rr_bp = bandpass_sos(xw, fs, 0.1, 0.5, order=4)
            out = fft_peak(rr_bp, fs, 0.1, 0.5)
            if out is not None:
                f_rr, snr_rr, prom_rr = out
                RR_rpm = 60.0 * f_rr
                print(f"RR={RR_rpm:5.1f} rpm | SNR={snr_rr:6.2f} | prom={prom_rr: .3e}")
            closeProgram.t_last_rr = now

        # ===== Cardiaque (HR) : 0.8–2.5 Hz, fenêtre 10s, update ~1s =====
        W_hr = int(10 * fs)
        if len(closeProgram.buf_all) >= W_hr and (now - closeProgram.t_last_hr) >= 1.0:
            xw = np.array(closeProgram.buf_all[-W_hr:], dtype=np.float64)
            hr_bp = bandpass_sos(xw, fs, 0.8, 2.5, order=4)
            out = fft_peak(hr_bp, fs, 0.8, 2.5)
            if out is not None:
                f_hr, snr_hr, prom_hr = out
                HR_bpm = 60.0 * f_hr
                print(f"HR={HR_bpm:5.1f} bpm | SNR={snr_hr:6.2f} | prom={prom_hr: .3e}")
            closeProgram.t_last_hr = now







        time.sleep(t)

except KeyboardInterrupt:
    closeProgram()

finally:
    print("Arrêt utilisateur")
