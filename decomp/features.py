"""Output audio features for the reference groupings: MFCC and chroma mean/variance, mean log-mel spectrum."""
import librosa
import numpy as np

SR = 48000


def audio_features(wav):
    y, sr = librosa.load(wav, sr=SR, mono=True)
    S = np.abs(librosa.stft(y))
    log_mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=sr))
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=13)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return {"mfcc": np.r_[mfcc.mean(1), mfcc.var(1)], "chroma": np.r_[chroma.mean(1), chroma.var(1)],
            "spectrogram": log_mel.mean(1)}


def write(model):
    d = model.dir / "outputs"
    rows = [audio_features(str(d / "wav" / f"{g}.wav")) for g in model.outputs().gen_id]
    out = model.dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "features.npz", **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
