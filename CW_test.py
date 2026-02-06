import numpy as np

def IQRawToComplex(I_12, Q_12, W=200, reset=False):
    """
    Traite une frame I/Q (arrays) et retourne un seul échantillon complexe utile.
    """
    I_12 = np.asarray(I_12, dtype=np.float64)
    Q_12 = np.asarray(Q_12, dtype=np.float64)

    # Sélection du bin le plus énergétique
    power = I_12**2 + Q_12**2
    idx = int(np.argmax(power))
    I_raw = I_12[idx]
    Q_raw = Q_12[idx]

    # Initialisation / reset
    if reset or not hasattr(IQRawToComplex, "bufI") or IQRawToComplex.bufI.size != W:
        IQRawToComplex.bufI = np.zeros(W)
        IQRawToComplex.bufQ = np.zeros(W)
        IQRawToComplex.sumI = 0.0
        IQRawToComplex.sumQ = 0.0
        IQRawToComplex.i = 0
        IQRawToComplex.n = 0

    # Recentrage 12 bits
    I0 = I_raw - 2047.5
    Q0 = Q_raw - 2047.5

    # Moyenne glissante temporelle
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

    return I_corr + 1j*Q_corr, I_corr, Q_corr, idx


import time

# Reset du filtre au démarrage
IQRawToComplex(0, 0, reset=True)

try:
    while True:

        # Exemple générique de récupération des données uRAD
        # À ADAPTER à ta fonction exacte du SDK
        return_code, _, _, _, I_12, Q_12, _ = uRAD_RP_SDK11.detection()

        if return_code != 0:
            print("Erreur radar")
            break

        signal, I_corr, Q_corr, idx = IQRawToComplex(I_12, Q_12)

        # === AFFICHAGE DES I/Q UTILES ===
        print(f"I = {I_corr:+8.3f} | Q = {Q_corr:+8.3f} | bin = {idx}")

        time.sleep(0.05)  # ~20 Hz (à adapter)

except KeyboardInterrupt:
    print("Arrêt utilisateur")
