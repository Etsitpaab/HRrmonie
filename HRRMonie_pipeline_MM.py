import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, get_window, find_peaks
import uRAD_RP_SDK11
import matplotlib.pyplot as plt
from collections import deque


# ============================================================
# Configuration radar
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
# Prétraitement IQ
# ============================================================

beta_dc_iq = 0.001
beta_var_iq = 0.001
eps = 1e-6

mean_I_iq = None
mean_Q_iq = None
var_I_iq = None
var_Q_iq = None


def pretraitement_affichage_iq(I, Q):

    global mean_I_iq, mean_Q_iq, var_I_iq, var_Q_iq

    if mean_I_iq is None:
        mean_I_iq = I
        mean_Q_iq = Q
        var_I_iq = 1
        var_Q_iq = 1

    mean_I_iq = (1-beta_dc_iq)*mean_I_iq + beta_dc_iq*I
    mean_Q_iq = (1-beta_dc_iq)*mean_Q_iq + beta_dc_iq*Q

    Ic = I - mean_I_iq
    Qc = Q - mean_Q_iq

    var_I_iq = (1-beta_var_iq)*var_I_iq + beta_var_iq*(Ic**2)
    var_Q_iq = (1-beta_var_iq)*var_Q_iq + beta_var_iq*(Qc**2)

    In = Ic/np.sqrt(var_I_iq+eps)
    Qn = Qc/np.sqrt(var_Q_iq+eps)

    return In, Qn


# ============================================================
# Prétraitement phase
# ============================================================

beta_dc_phase = 0.01
mean_I_phase = None
mean_Q_phase = None


def pretraitement_phase(I,Q):

    global mean_I_phase, mean_Q_phase

    if mean_I_phase is None:
        mean_I_phase = I
        mean_Q_phase = Q

    mean_I_phase = (1-beta_dc_phase)*mean_I_phase + beta_dc_phase*I
    mean_Q_phase = (1-beta_dc_phase)*mean_Q_phase + beta_dc_phase*Q

    return I-mean_I_phase , Q-mean_Q_phase


# ============================================================
# Filtres
# ============================================================

def filtre_pass_haut(signal,fs,fc=0.05,ordre=4):

    nyq = fs/2
    sos = butter(ordre,fc/nyq,btype='highpass',output='sos')
    return sosfiltfilt(sos,signal)


def passe_bande(signal,f_low,f_high,fs,ordre=4):

    nyq = fs/2
    sos = butter(ordre,[f_low/nyq,f_high/nyq],btype='bandpass',output='sos')
    return sosfiltfilt(sos,signal)


# ============================================================
# FFT
# ============================================================

def spectre_fft(x,fs):

    N=len(x)
    nfft=int(2**np.ceil(np.log2(N)))

    w=get_window("hann",N)
    xw=(x-np.mean(x))*w

    X=np.fft.rfft(xw,nfft)

    P=np.abs(X)**2
    f=np.fft.rfftfreq(nfft,1/fs)

    return f,P


# ============================================================
# Extraction pics spectraux
# ============================================================

def extraire_pics(signal,fs,fmin,fmax):

    f,P = spectre_fft(signal,fs)

    mask=(f>=fmin)&(f<=fmax)

    f_band=f[mask]
    P_band=P[mask]

    peaks,_=find_peaks(P_band)

    if len(peaks)==0:
        peaks=[np.argmax(P_band)]

    candidats=[]

    bruit=np.median(P_band)+1e-12

    for k in peaks:

        freq=f_band[k]
        power=P_band[k]
        snr=10*np.log10(power/bruit)

        candidats.append({
            "freq":freq,
            "power":power,
            "snr":snr
        })

    candidats=sorted(candidats,key=lambda x:x["power"],reverse=True)

    return candidats


# ============================================================
# Choix HR dynamique
# ============================================================

def choisir_hr_dynamique(candidats,rr,historique):

    if len(candidats)==0:
        return np.nan

    pmax=max(c["power"] for c in candidats)

    prev=historique[-1] if len(historique)>0 else np.nan
    med=np.median(historique) if len(historique)>0 else np.nan

    best_freq=np.nan
    best_score=-1

    for c in candidats:

        f=c["freq"]
        p=c["power"]

        score_p=p/pmax

        if np.isfinite(rr):

            d=min(abs(f-n*rr) for n in range(2,6))
            score_h=min(1,d/0.08)

        else:
            score_h=1

        if np.isfinite(prev):
            score_t=np.exp(-(f-prev)**2/(2*0.2**2))
        else:
            score_t=1

        if np.isfinite(med):
            score_s=np.exp(-(f-med)**2/(2*0.25**2))
        else:
            score_s=1

        score = (
            0.35*score_p +
            0.25*score_h +
            0.25*score_t +
            0.15*score_s
        )

        if score>best_score:
            best_score=score
            best_freq=f

    if best_score<0.35:
        return np.nan

    return best_freq


# ============================================================
# Affichage constellation
# ============================================================

plt.ion()
fig,ax=plt.subplots()
sc=ax.scatter([],[],s=6)

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)

histo_i=deque(maxlen=4000)
histo_q=deque(maxlen=4000)


# ============================================================
# Buffers
# ============================================================

buffer_t=deque()
buffer_phi=deque()

historique_hr=deque(maxlen=8)

fs_est=None
beta_fs=0.05

last_print=time.time()

fenetre_signal=30
echantillons_min=200


# ============================================================
# Lock range bin
# ============================================================

idx_lock=None


def selection_bin(z_bin):

    global idx_lock

    amp=np.abs(z_bin)

    if idx_lock is None:
        idx_lock=np.argmax(amp)

    left=max(0,idx_lock-2)
    right=min(len(z_bin),idx_lock+3)

    idx_lock=left+np.argmax(amp[left:right])

    z=np.mean(z_bin[max(0,idx_lock-1):idx_lock+2])

    return z,idx_lock


# ============================================================
# Boucle principale
# ============================================================

try:

    while True:

        err,res,IQ=uRAD_RP_SDK11.detection()

        if err!=0:
            closeProgram()

        I=np.array(IQ[0],dtype=float)
        Q=np.array(IQ[1],dtype=float)

        z_bin=I+1j*Q

        z_sel,idx=selection_bin(z_bin)

        I_sel=np.real(z_sel)
        Q_sel=np.imag(z_sel)

        # constellation

        I_aff,Q_aff=pretraitement_affichage_iq(I_sel,Q_sel)

        z_aff=I_aff+1j*Q_aff

        if abs(z_aff)>0:
            z_aff/=abs(z_aff)

        histo_i.append(np.real(z_aff))
        histo_q.append(np.imag(z_aff))

        sc.set_offsets(np.c_[histo_i,histo_q])

        plt.pause(0.001)

        # phase

        I_c,Q_c=pretraitement_phase(I_sel,Q_sel)

        phase=np.arctan2(Q_c,I_c)

        t=time.time()

        buffer_t.append(t)
        buffer_phi.append(phase)

        if len(buffer_t)>=2:

            dt=buffer_t[-1]-buffer_t[-2]
            fs_inst=1/max(dt,1e-6)

            if fs_est is None:
                fs_est=fs_inst
            else:
                fs_est=(1-beta_fs)*fs_est+beta_fs*fs_inst

        while buffer_t[-1]-buffer_t[0]>fenetre_signal:

            buffer_t.popleft()
            buffer_phi.popleft()

        if len(buffer_phi)>=echantillons_min and fs_est is not None:

            phi=np.unwrap(np.array(buffer_phi))

            phi_hp=filtre_pass_haut(phi,fs_est)

            sig_rr=passe_bande(phi_hp,0.10,0.50,fs_est)
            sig_hr=passe_bande(phi_hp,0.80,2.50,fs_est)

            rr_candidats=extraire_pics(sig_rr,fs_est,0.10,0.50)

            rr=rr_candidats[0]["freq"] if len(rr_candidats)>0 else np.nan

            hr_candidats=extraire_pics(sig_hr,fs_est,0.80,2.50)

            hr=choisir_hr_dynamique(hr_candidats,rr,historique_hr)

            if np.isfinite(hr):
                historique_hr.append(hr)

            rr_rpm=rr*60 if np.isfinite(rr) else np.nan
            hr_bpm=hr*60 if np.isfinite(hr) else np.nan

            if time.time()-last_print>1:

                print(
                    f"\nIDX {idx}"
                    f"\nRR : {rr_rpm:.2f} rpm"
                    f"\nHR : {hr_bpm:.2f} bpm"
                    f"\nfs : {fs_est:.2f} Hz"
                )

                last_print=time.time()

except KeyboardInterrupt:

    closeProgram()