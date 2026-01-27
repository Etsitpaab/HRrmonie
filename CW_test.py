# cw_mode1_thorax.py
# Requiert: Raspberry Pi + spidev + gpiozero + numpy
# pip install numpy spidev gpiozero

import time
import numpy as np
import uRAD_RP_SDK11 as urad

def robust_center_12bit(x):
    """
    Le SDK reconstruit I/Q en 12-bit non signé (0..4095) :contentReference[oaicite:1]{index=1}.
    En pratique, I/Q sont centrés ~2048. On recentre pour faire une phase stable.
    """
    x = np.asarray(x, dtype=np.float64)
    return x - 2048.0

def unwrap_phase(ph):
    """Unwrap simple (évite les sauts ±pi)."""
    return np.unwrap(ph)

def phase_to_displacement_m(ph_unwrapped, fc_hz):
    """
    Radar monostatique:
      phase = 4πR/λ  =>  ΔR = (λ / (4π)) * Δphase
    """
    c = 299_792_458.0
    lam = c / fc_hz
    dphi = ph_unwrapped - ph_unwrapped[0]
    dR = (lam / (4.0 * np.pi)) * dphi
    return dR  # en mètres

def main():
    # ======================
    # 1) Configuration CW
    # ======================
    mode = 1  # CW :contentReference[oaicite:2]{index=2}

    # IMPORTANT: Dans ce SDK, f0 est borné [5..245] en CW :contentReference[oaicite:3]{index=3}.
    # Sans ton manuel, on ne sait pas l’unité exacte (souvent: GHz*10 ou MHz*??).
    # Mets ici la valeur correspondant à ~24 GHz selon ton manuel.
    # Exemple plausible: 240 -> 24.0 GHz si f0 est en 0.1 GHz.
    f0 = 240

    BW = 100   # ignoré/peu pertinent en CW, mais le SDK impose un BW min/max :contentReference[oaicite:4]{index=4}
    Ns = 200   # nombre d’échantillons I/Q renvoyés (voir calcul Ns_temp dans detection) :contentReference[oaicite:5]{index=5}
    Ntar = 1   # 1 cible principale
    Rmax = 20  # en mode 1, le SDK le traite comme un "VmaxMax" (limite) :contentReference[oaicite:6]{index=6}
    MTI = 0
    Mth = 4
    Alpha = 10

    # On demande distance + I/Q pour:
    # - distance: seulement si dispo en CW côté firmware
    # - I/Q: nécessaire au micro-mouvement
    distance_true = True
    velocity_true = False
    SNR_true = True
    I_true = True
    Q_true = True
    movement_true = False

    ret = urad.loadConfiguration(
        mode, f0, BW, Ns, Ntar, Rmax, MTI, Mth, Alpha,
        distance_true, velocity_true, SNR_true, I_true, Q_true, movement_true
    )
    if ret != 0:
        raise RuntimeError(f"loadConfiguration() a échoué: code {ret}")

    ret = urad.turnON()
    if ret != 0:
        raise RuntimeError(f"turnON() a échoué: code {ret}")

    # ======================
    # 2) Boucle acquisition
    # ======================
    # Fréquence porteuse pour convertir phase->déplacement.
    # À adapter à TON paramétrage réel.
    # Si f0=240 correspond à 24.0 GHz => fc_hz=24e9
    fc_hz = 24.0e9

    # On collecte un buffer temporel (ex: 10 s à ~50 Hz)
    duration_s = 10.0
    target_fs = 50.0
    dt = 1.0 / target_fs
    n = int(duration_s * target_fs)

    disp = []
    dist_coarse = []
    snr_list = []

    t0 = time.time()
    prev_phase0 = None
    phase0_unwrapped_offset = 0.0
    last_phase0 = None

    for k in range(n):
        # Lance une "detection" + récupère results + I/Q :contentReference[oaicite:7]{index=7}
        code, results, iq = urad.detection()
        if code != 0:
            # erreurs typiques: -1..-6 (timeout, longueur, etc.) :contentReference[oaicite:8]{index=8}
            print(f"[WARN] detection() code={code}")
            time.sleep(dt)
            continue

        # results = [NtarDetected, distance[], velocity[], SNR[], movement] :contentReference[oaicite:9]{index=9}
        ntar_detected, distance_arr, _, snr_arr, _ = results
        I, Q = iq

        # 2.1) Distance "coarse" (si le firmware la renvoie en CW)
        # On prend la cible 0 si SNR > 0
        if isinstance(distance_arr, list) and len(distance_arr) > 0:
            d0 = distance_arr[0]
            dist_coarse.append(d0)
        if isinstance(snr_arr, list) and len(snr_arr) > 0:
            snr_list.append(snr_arr[0])

        # 2.2) Phase et micro-déplacement
        I0 = robust_center_12bit(I)
        Q0 = robust_center_12bit(Q)

        # Phase instantanée par échantillon
        ph = np.arctan2(Q0, I0)

        # Pour un signal “thorax”, une pratique simple:
        # prendre la phase moyenne (ou médiane) sur le burst pour réduire le bruit.
        ph0 = float(np.median(ph))

        # Unwrap manuel dans le temps (sur ph0 à chaque frame)
        if last_phase0 is None:
            last_phase0 = ph0
            ph0_unwrapped = ph0
        else:
            d = ph0 - last_phase0
            # ramener dans [-pi, pi]
            while d > np.pi:
                d -= 2*np.pi
            while d < -np.pi:
                d += 2*np.pi
            phase0_unwrapped_offset += d
            last_phase0 = ph0
            ph0_unwrapped = phase0_unwrapped_offset

        # Convertit phase->déplacement (relatif) autour de l’instant initial
        # ici on intègre Δphase au fil du temps => déplacement relatif
        # (comme ph0_unwrapped est déjà un cumul, on l’utilise directement)
        dR = phase_to_displacement_m(np.array([0.0, ph0_unwrapped]), fc_hz)[-1]
        disp.append(dR)

        # pacing
        elapsed = time.time() - t0
        next_t = (k + 1) * dt
        sleep_t = next_t - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    urad.turnOFF()

    disp = np.asarray(disp, dtype=np.float64)
    if disp.size == 0:
        print("Aucune donnée de déplacement exploitable.")
        return

    # ======================
    # 3) Post-traitement simple
    # ======================
    # On retire la dérive (DC) pour isoler respiration/rythme cardiaque.
    disp_detrended = disp - np.mean(disp)

    # Estimation grossière respiration/FC via FFT (sans filtrage sophistiqué)
    fs = target_fs
    N = disp_detrended.size
    freqs = np.fft.rfftfreq(N, d=1/fs)
    spec = np.abs(np.fft.rfft(disp_detrended))

    # Fenêtres typiques:
    # respiration: ~0.1–0.6 Hz (6–36 bpm)
    # coeur: ~0.8–3.0 Hz (48–180 bpm)
    def peak_in_band(fmin, fmax):
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if idx.size == 0:
            return None
        i = idx[np.argmax(spec[idx])]
        return freqs[i]

    f_resp = peak_in_band(0.10, 0.60)
    f_hr   = peak_in_band(0.80, 3.00)

    if f_resp is not None:
        print(f"Respiration estimée: {f_resp*60:.1f} cycles/min")
    else:
        print("Respiration: non détectée")

    if f_hr is not None:
        print(f"Fréquence cardiaque estimée: {f_hr*60:.1f} bpm")
    else:
        print("FC: non détectée")

    # Distance coarse (si dispo)
    if len(dist_coarse) > 0:
        dc = np.asarray(dist_coarse, dtype=np.float64)
        print(f"Distance (coarse, si fournie par le firmware): médiane={np.median(dc):.2f} (unités du module)")

    # Micro-mouvement (amplitude)
    print(f"Déplacement thorax relatif (RMS): {np.sqrt(np.mean(disp_detrended**2))*1e3:.2f} mm")
    print(f"Déplacement thorax relatif (peak-to-peak): {(np.max(disp_detrended)-np.min(disp_detrended))*1e3:.2f} mm")

if __name__ == "__main__":
    main()