"""Objective AV-offset estimator (AB-A-0027 finding 1).

Speech drives mouth motion, so per-frame VIDEO MOTION energy should track the AUDIO ENVELOPE.
Cross-correlate the two over +-500 ms and report the lag that maximises correlation.

Sign convention: POSITIVE lag = video motion happens LATER than the audio (video lags; the fix is
to delay the audio by that much, i.e. the operator's "audio-later" rung).

This replaces a human ladder with a number, and applies identically to any lane, so ours, the
vendor's and the cloud's are directly comparable.
"""
import subprocess, sys, numpy as np

def video_motion(path, fps=24.0, w=96, h=64):
    out = subprocess.run(["ffmpeg","-v","error","-i",path,"-vf",f"scale={w}:{h},fps={fps}",
                          "-f","rawvideo","-pix_fmt","gray","-"], capture_output=True, check=True).stdout
    f = np.frombuffer(out, dtype=np.uint8).astype(np.float64).reshape(-1, h*w)
    d = np.abs(np.diff(f, axis=0)).mean(axis=1)          # (N-1,)
    return d

def audio_env(path, fps=24.0, sr=16000):
    out = subprocess.run(["ffmpeg","-v","error","-i",path,"-ac","1","-ar",str(sr),
                          "-f","f32le","-"], capture_output=True, check=True).stdout
    x = np.frombuffer(out, dtype=np.float32).astype(np.float64)
    hop = int(sr/fps)
    n = len(x)//hop
    env = np.array([np.sqrt((x[i*hop:(i+1)*hop]**2).mean()+1e-12) for i in range(n)])
    return np.diff(env)                                   # match motion's differencing

def z(a):
    a = a - a.mean()
    s = a.std()
    return a/s if s > 0 else a

def offset_ms(path, fps=24.0, max_ms=500):
    m, e = video_motion(path, fps), audio_env(path, fps)
    n = min(len(m), len(e)); m, e = z(m[:n]), z(e[:n])
    maxlag = int(round(max_ms/1000*fps))
    best, bl = -9, 0
    for lag in range(-maxlag, maxlag+1):
        # positive lag: motion delayed vs audio
        if lag >= 0: a, b = m[lag:], e[:n-lag]
        else:        a, b = m[:n+lag], e[-lag:]
        if len(a) < fps: continue
        # ⚠️ Pearson r on THIS overlap. z-scoring the full arrays and then slicing (the first
        # draft) leaves each lag normalised by the wrong mean/σ, which biases the argmax — it
        # failed validation on a ladder with known 200 ms steps.
        aa, bb = a - a.mean(), b - b.mean()
        d = (aa.std()*bb.std())
        c = float((aa*bb).mean()/d) if d > 0 else -9
        if c > best: best, bl = c, lag
    return bl*1000.0/fps, best

for p in sys.argv[1:]:
    ms, c = offset_ms(p)
    print(f"  {ms:+7.0f} ms   r={c:.3f}   {p.split('/')[-1]}")
