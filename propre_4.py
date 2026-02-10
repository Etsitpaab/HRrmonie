import time
import math
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
import uRAD_RP_SDK11
import matplotlib.pyplot as plt


# Configuration radar uRAD (inchangée)

mode = 3
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

# Paramètres acquisition

t = 0.05
fs = 1.0 / t

c = 299_792_458.0
fc = 24.005e9
lam = c / fc


def closeProgram():
    uRAD_RP_SDK11.turnOFF()
    raise SystemExit

# Init radar (inchangée)

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

# 1) Prétraitement IQ : bin max énergie + retrait DC (moy gliss)

def iq_preprocess_bestbin(I_12, Q_12, W=200, reset=False):
    I_12 = np.asarray(I_12, dtype=np.float64)
    Q_12 = np.asarray(Q_12, dtype=np.float64)

    p = I_12**2 + Q_12**2
    idx = int(np.argmax(p))

    I0 = float(I_12[idx]) - 2047.5
    Q0 = float(Q_12[idx]) - 2047.5

    if reset or not hasattr(iq_preprocess_bestbin, "bufI") or iq_preprocess_bestbin.bufI.size != W:
        iq_preprocess_bestbin.bufI = np.zeros(W, dtype=np.float64)
        iq_preprocess_bestbin.bufQ = np.zeros(W, dtype=np.float64)
        iq_preprocess_bestbin.sumI = 0.0
        iq_preprocess_bestbin.sumQ = 0.0
        iq_preprocess_bestbin.i = 0
        iq_preprocess_bestbin.n = 0

    j = iq_preprocess_bestbin.i
    iq_preprocess_bestbin.sumI += I0 - iq_preprocess_bestbin.bufI[j]
    iq_preprocess_bestbin.sumQ += Q0 - iq_preprocess_bestbin.bufQ[j]
    iq_preprocess_bestbin.bufI[j] = I0
    iq_preprocess_bestbin.bufQ[j] = Q0

    iq_preprocess_bestbin.i = (j + 1) % W
    iq_preprocess_bestbin.n = min(iq_preprocess_bestbin.n + 1, W)

    I_dc = iq_preprocess_bestbin.sumI / iq_preprocess_bestbin.n
    Q_dc = iq_preprocess_bestbin.sumQ / iq_preprocess_bestbin.n

    I_corr = I0 - I_dc
    Q_corr = Q0 - Q_dc
    return float(I_corr), float(Q_corr), idx

# 2) Unwrap incrémental de la phase

def phase_unwrap_incremental(phi, reset=False):
    phi = float(phi)
    if reset or not hasattr(phase_unwrap_incremental, "prev"):
        phase_unwrap_incremental.prev = phi
        phase_unwrap_incremental.offset = 0.0
        return phi

    d = phi - phase_unwrap_incremental.prev
    if d > np.pi:
        phase_unwrap_incremental.offset -= 2.0 * np.pi
    elif d < -np.pi:
        phase_unwrap_incremental.offset += 2.0 * np.pi

    phase_unwrap_incremental.prev = phi
    return phi + phase_unwrap_incremental.offset



# 3) Suppression dérive & artefacts lents (baseline + gate)
#    -> renvoie x_hp (m) + valid

def drift_suppress_and_gate(x_m, A, fs, tau_base=10.0, tau_amp=5.0, var_win_s=2.0, var_thr=4e-6, reset=False):
    x_m = float(x_m)
    A = float(A)

    if reset or not hasattr(drift_suppress_and_gate, "init"):
        drift_suppress_and_gate.init = True
        drift_suppress_and_gate.alpha0 = 1.0 / max(1.0, tau_base * fs)
        drift_suppress_and_gate.base = x_m

        drift_suppress_and_gate.betaA = 1.0 / max(1.0, tau_amp * fs)
        drift_suppress_and_gate.A_mu = A
        drift_suppress_and_gate.A_var = 0.0

        drift_suppress_and_gate.Wvar = int(max(10, round(var_win_s * fs)))
        drift_suppress_and_gate.buf = np.zeros(drift_suppress_and_gate.Wvar, dtype=np.float64)
        drift_suppress_and_gate.k = 0
        drift_suppress_and_gate.nv = 0

    dA = A - drift_suppress_and_gate.A_mu
    drift_suppress_and_gate.A_mu += drift_suppress_and_gate.betaA * dA
    drift_suppress_and_gate.A_var = (1.0 - drift_suppress_and_gate.betaA) * drift_suppress_and_gate.A_var + drift_suppress_and_gate.betaA * (dA * dA)
    A_sigma = math.sqrt(max(1e-12, drift_suppress_and_gate.A_var))

    A_thr = max(drift_suppress_and_gate.A_mu - 3.0 * A_sigma, 1e-6)
    valid_amp = (A >= A_thr)

    w = 0.0
    if valid_amp:
        w = (A - A_thr) / max(1e-6, drift_suppress_and_gate.A_mu - A_thr)
        w = min(max(w, 0.0), 1.0)

    alpha = drift_suppress_and_gate.alpha0 * w
    drift_suppress_and_gate.base = (1.0 - alpha) * drift_suppress_and_gate.base + alpha * x_m
    x_hp = x_m - drift_suppress_and_gate.base

    j = drift_suppress_and_gate.k
    drift_suppress_and_gate.buf[j] = x_hp
    drift_suppress_and_gate.k = (j + 1) % drift_suppress_and_gate.Wvar
    drift_suppress_and_gate.nv = min(drift_suppress_and_gate.nv + 1, drift_suppress_and_gate.Wvar)

    movement = False
    if drift_suppress_and_gate.nv >= int(fs):
        v = float(np.var(drift_suppress_and_gate.buf[:drift_suppress_and_gate.nv]))
        movement = (v > var_thr)

    valid = valid_amp and (not movement)
    return float(x_hp), bool(valid)

# 4) Filtrage passe-bande (RR/HR) utilitaire

def bandpass_sos(x, fs, f1, f2, order=4):
    sos = butter(order, [f1/(fs/2), f2/(fs/2)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)

# 5) Estimation fréquence par FFT + métriques

def estimate_rate_fft(x, fs, fmin, fmax):
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

    peaks, props = find_peaks(P2, prominence=np.max(P2) * 0.02)
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
    return peak_f, float(snr), float(prom)

# 6) Buffer + estimations RR/HR (tout en fonctions, pas de classe)

def buffer_push_xhp(x_hp, valid, fs, keep_s=40.0, reset=False):
    if reset or not hasattr(buffer_push_xhp, "buf"):
        buffer_push_xhp.buf = []
        buffer_push_xhp.max_keep = int(keep_s * fs)

    if valid:
        buffer_push_xhp.buf.append(float(x_hp))
        if len(buffer_push_xhp.buf) > buffer_push_xhp.max_keep:
            buffer_push_xhp.buf = buffer_push_xhp.buf[-buffer_push_xhp.max_keep:]

    return buffer_push_xhp.buf

def rr_estimate_from_buffer(fs, window_s=20.0, update_s=1.0, reset=False):
    now = time.time()
    if reset or not hasattr(rr_estimate_from_buffer, "t_last"):
        rr_estimate_from_buffer.t_last = 0.0

    W = int(window_s * fs)
    buf = buffer_push_xhp.buf if hasattr(buffer_push_xhp, "buf") else []

    if len(buf) < W or (now - rr_estimate_from_buffer.t_last) < update_s:
        return None

    xw = np.array(buf[-W:], dtype=np.float64)
    rr_bp = bandpass_sos(xw, fs, 0.1, 0.5, order=4)
    out = estimate_rate_fft(rr_bp, fs, 0.1, 0.5)

    rr_estimate_from_buffer.t_last = now
    if out is None:
        return None

    f_rr, snr, prom = out
    return 60.0 * f_rr, snr, prom

def hr_estimate_from_buffer(fs, window_s=10.0, update_s=1.0, reset=False):
    now = time.time()
    if reset or not hasattr(hr_estimate_from_buffer, "t_last"):
        hr_estimate_from_buffer.t_last = 0.0

    W = int(window_s * fs)
    buf = buffer_push_xhp.buf if hasattr(buffer_push_xhp, "buf") else []

    if len(buf) < W or (now - hr_estimate_from_buffer.t_last) < update_s:
        return None

    xw = np.array(buf[-W:], dtype=np.float64)
    hr_bp = bandpass_sos(xw, fs, 0.8, 2.5, order=4)
    out = estimate_rate_fft(hr_bp, fs, 0.8, 2.5)

    hr_estimate_from_buffer.t_last = now
    if out is None:
        return None

    f_hr, snr, prom = out
    return 60.0 * f_hr, snr, prom

def init_live_plot(fs, window_s=30.0):
    plt.ion()

    fig, ax = plt.subplots(figsize=(10, 4))

    N = int(window_s * fs)
    t = np.linspace(-window_s, 0, N)

    line, = ax.plot(t, np.zeros(N))

    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Déplacement (mm)")
    ax.set_title("Déplacement thoracique (x_hp)")
    ax.grid(True)

    ax.set_ylim(-5, 5)   # ±5 mm (à ajuster)
    ax.set_xlim(-window_s, 0)

    fig.tight_layout()

    return {
        "fig": fig,
        "ax": ax,
        "line": line,
        "t": t,
        "N": N
    }
 
plot_ctx = init_live_plot(fs, window_s=30.0)
    
def update_live_plot(plot_ctx, x_hp_m):
    # buffer interne
    if not hasattr(update_live_plot, "buf"):
        update_live_plot.buf = np.zeros(plot_ctx["N"])

    # shift gauche + ajout nouvelle valeur
    update_live_plot.buf[:-1] = update_live_plot.buf[1:]
    update_live_plot.buf[-1] = x_hp_m * 1e3  # m → mm

    plot_ctx["line"].set_ydata(update_live_plot.buf)
    plot_ctx["fig"].canvas.draw()
    plot_ctx["fig"].canvas.flush_events()



# Boucle principale (main)

try:
    while True:
        return_code, results, raw_results = uRAD_RP_SDK11.detection()
        if return_code != 0:
            print("Erreur radar")
            break

        I = raw_results[0]
        Q = raw_results[1]

        I_corr, Q_corr, idx = iq_preprocess_bestbin(I, Q, W=200)

        A = math.sqrt(I_corr*I_corr + Q_corr*Q_corr)
        phi = math.atan2(Q_corr, I_corr)
        phi_u = phase_unwrap_incremental(phi)

        x_m = (lam / (4.0 * math.pi)) * phi_u
        x_hp, valid = drift_suppress_and_gate(x_m, A, fs)

        buffer_push_xhp(x_hp, valid, fs, keep_s=40.0)

        rr = rr_estimate_from_buffer(fs, window_s=20.0, update_s=1.0)
        hr = hr_estimate_from_buffer(fs, window_s=10.0, update_s=1.0)

        print(f"bin={idx:3d} | I={I_corr:+8.3f} Q={Q_corr:+8.3f} | phi_u={phi_u:+8.3f} | x_hp(mm)={x_hp*1e3:+7.3f} | valid={valid}")
       
        x_hp, valid = drift_suppress_and_gate(x_m, A, fs)

        if rr is not None:
            RR_rpm, snr_rr, prom_rr = rr
            print(f"RR={RR_rpm:5.1f} rpm | SNR={snr_rr:6.2f} | prom={prom_rr:.3e}")

        if hr is not None:
            HR_bpm, snr_hr, prom_hr = hr
            print(f"HR={HR_bpm:5.1f} bpm | SNR={snr_hr:6.2f} | prom={prom_hr:.3e}")

        if valid:
            update_live_plot(plot_ctx, x_hp)

        time.sleep(t)

except KeyboardInterrupt:
    closeProgram()

finally:
    print("Arrêt utilisateur")
