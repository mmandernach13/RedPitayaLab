import time
import numpy as np
import matplotlib.pyplot as plt
from pyrpl import Pyrpl

f = 100   # reference frequency [Hz]
mdm_f = 50000 #iq module modulation/demodulation frequency
N_FFT_SHOW = 8

<<<<<<< HEAD
rp = Pyrpl(config="mason_config", hostname="169.254.148.168", modules={})
=======
rp = Pyrpl(config="", hostname="169.254.131.37", modules={})
>>>>>>> e4a527fdfcc156da9befceb9b2c0f3f73525a8bd

# Modules
asg = rp.rp.asg0
iq = rp.rp.iq0
scope = rp.rp.scope

# Generate reference signal on out1
print("initializng asg")
asg.output_direct = 'out1'
asg.setup(waveform='sin',
          frequency=f,
          amplitude=0.8,
          offset=0,
          trigger_source='immediately')

# Setup lock-in
print("initializing iq")
iq.setup(frequency=mdm_f,
         bandwidth=[f, f*1.25],   # Hz
         gain=0.0,
         phase=0,
         acbandwidth=f*0.75,
         amplitude=0.9,
         input='in1',
         output_direct='out2',
         output_signal='quadrature',
         quadrature_factor=10)

# Setup scope
print("initializing scope")
scope.setup(input1='out1', input2='iq0', decimation=16384)
scope._start_acquisition_rolling_mode()

# -----------------------
# single capture helper
# -----------------------
def single_capture():
    scope.single()
    ch1 = np.array(scope._data_ch1_current)   # reference (out1)
    ch2 = np.array(scope._data_ch2_current)   # input (in1)
    return ch1, ch2

# -----------------------
# estimate sample rate from zero-crossings of reference
# -----------------------
def estimate_sr_from_ref(ref_signal, known_freq):
    zc = np.where((ref_signal[:-1] < 0) & (ref_signal[1:] >= 0))[0]
    if len(zc) < 2:
        return None
    periods = np.diff(zc)
    median_samples_per_period = float(np.median(periods))
    sr_est = median_samples_per_period * known_freq
    return sr_est

# -----------------------
# software demodulation
# -----------------------
def soft_lockin(sig, ref, f, sr):
    N = len(sig)
    t = np.arange(N) / sr
<<<<<<< HEAD
    cos_ref = np.cos(2*np.pi*20*f*t)
    sin_ref = np.sin(2*np.pi*20*f*t)
=======
    cos_ref = np.cos(2*np.pi*f*t)
    sin_ref = np.sin(2*np.pi*f*t)
>>>>>>> e4a527fdfcc156da9befceb9b2c0f3f73525a8bd
    # multiply and average (2*mean gives amplitude for a pure tone)
    X = 2.0 * np.mean(sig * cos_ref)
    Y = 2.0 * np.mean(sig * sin_ref)
    R = np.sqrt(X*X + Y*Y)
    phi = np.degrees(np.arctan2(Y, X))
    return X, Y, R, phi, t, cos_ref, sin_ref

# -----------------------
# Capture, estimate SR, run soft lock-in, show diagnostics
# -----------------------
ch1, ch2 = single_capture()
print("Raw capture lengths:", len(ch1), len(ch2))

# estimate sampling rate
sr = estimate_sr_from_ref(ch1, f)
if sr is None:
    # fallback: try common scope attribute names
    for attr in ("sample_rate","sampling_rate","samplerate","sr","_sr"):
        if hasattr(scope, attr):
            try:
                sr = getattr(scope, attr)
                if callable(sr): sr = sr()
            except Exception:
                sr = None
        if sr: break

if sr is None:
    print("ERROR: couldn't estimate scope sample rate. Make sure the generator is producing a sine and wiring is correct.")
    raise SystemExit

print("Estimated sample rate: " + str(sr) + " Hz (samples/sec)")
t = np.linspace(0, len(ch2)/sr, len(ch2))
print("Captured " + str(t[len(ch2)-1]) + " seconds of data")

# run software lock-in
X, Y, R, phi, t, cos_ref, sin_ref = soft_lockin(ch2, ch1, f, sr)
print("\nSoftware lock-in results:")
print("X= " + str(X))
print("Y= " + str(Y))
print("R= " + str(R))

# FFT diagnostics of CH2
N = len(ch2)
fft = np.fft.rfft(ch2 * np.hanning(N))
freqs = np.fft.rfftfreq(N, 1.0/sr)
mag = np.abs(fft)
idx_sorted = np.argsort(mag)[::-1]
print("\nTop FFT peaks (freq [Hz], magnitude):")
for i in range(min(N_FFT_SHOW, len(idx_sorted))):
    ii = idx_sorted[i]
    print(str(i+1) + " : " + str(freqs[ii]) + " Hz : " + str(mag[ii]))

# -----------------------
# Plot traces and multiplied signals
# -----------------------
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t, ch1, label='CH1 (ref)')
plt.plot(t, ch2, label='CH2 (iq0)', alpha=0.8)
plt.legend()
plt.title("Raw traces (time domain)")

plt.subplot(3,1,2)
plt.plot(t, ch2 * cos_ref, label='iq0 * cos(ref)')
plt.plot(t, ch2 * sin_ref, label='iq0 * sin(ref)', alpha=0.8)
plt.legend()
plt.title("Instantaneous multiplications (before averaging)")

plt.subplot(3,1,3)
plt.semilogy(freqs, mag + 1e-12)
plt.xlim(0, 5*f)  # zoom near reference
plt.title("FFT of CH2 (zoomed near reference)")
plt.tight_layout()
plt.show()