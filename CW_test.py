import time
import uRAD_RP_SDK11 as urad

def main():
    # ====== Configuration CW (mode 1) ======
    mode = 1  # CW :contentReference[oaicite:1]{index=1}

    # f0 doit être cohérent avec ton manuel (ex: 240 ~ 24 GHz selon l'échelle du fabricant)
    f0 = 240

    BW = 100     # non pertinent en CW, mais requis par l'API :contentReference[oaicite:2]{index=2}
    Ns = 50      # petit Ns = boucle plus légère
    Ntar = 1     # 1 seule cible
    Rmax = 20    # en CW, ce paramètre est traité côté SDK comme une limite VmaxMax :contentReference[oaicite:3]{index=3}
    MTI = 0
    Mth = 4
    Alpha = 10

    distance_true = True
    velocity_true = False
    SNR_true = True
    I_true = False
    Q_true = False
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

    print("Acquisition CW pendant 60s... (CTRL+C pour arrêter)")
    t_end = time.time() + 60.0

    try:
        while time.time() < t_end:
            code, results, _ = urad.detection()  # results = [NtarDetected, distance[], velocity[], SNR[], movement] :contentReference[oaicite:4]{index=4}
            if code != 0:
                print(f"[WARN] detection() code={code}")
                time.sleep(0.05)
                continue

            ntar_detected, distance_arr, _, snr_arr, _ = results

            # distance_arr est une liste (même si Ntar=1)
            if isinstance(distance_arr, list) and len(distance_arr) > 0:
                d0 = distance_arr[0]
                snr0 = snr_arr[0] if isinstance(snr_arr, list) and len(snr_arr) > 0 else None
                ts = time.time()
                if snr0 is None:
                    print(f"{ts:.3f}  distance={d0}")
                else:
                    print(f"{ts:.3f}  distance={d0}  SNR={snr0}")
            else:
                print(f"{time.time():.3f}  distance=NA (pas renvoyée en CW)")

            time.sleep(0.05)  # ~20 Hz
    finally:
        urad.turnOFF()

if __name__ == "__main__":
    main()