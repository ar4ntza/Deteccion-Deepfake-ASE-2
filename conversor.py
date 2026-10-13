# script para convertir audios en espectrogramas

import os
import numpy as np
import librosa
from multiprocessing import Pool
from tqdm import tqdm

CARPETA_ENTRADA = "audiosEsp/fragmentos"
CARPETA_SALIDA = "melEsp"

SR = 22050       # usa 44100 si tus originales lo permiten
N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 20
FMAX = SR // 2
DURACION = 10    # segundos, largo fijo para todos

LARGO = int(SR * DURACION)
EXTENSIONES = (".wav", ".mp3", ".m4a", ".ogg", ".flac")

def procesar(nombre):
    base = os.path.splitext(nombre)[0]
    salida = os.path.join(CARPETA_SALIDA, base + ".npy")
    if os.path.exists(salida):
        return None
    try:
        y, _ = librosa.load(os.path.join(CARPETA_ENTRADA, nombre), sr=SR, mono=True)

        # Largo fijo: rellena con ceros o recorta
        if len(y) < LARGO:
            y = np.pad(y, (0, LARGO - len(y)))
        else:
            y = y[:LARGO]

        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=1.0, top_db=None).astype(np.float32)
        np.save(salida, mel_db)
        return None
    except Exception as e:
        return f"{nombre}: {e}"

if __name__ == "__main__":
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    archivos = [f for f in os.listdir(CARPETA_ENTRADA) if f.lower().endswith(EXTENSIONES)]

    with Pool(processes=max(1, os.cpu_count() - 1)) as pool:
        errores = [r for r in tqdm(pool.imap_unordered(procesar, archivos, chunksize=16),
                                   total=len(archivos)) if r]

    print(f"Terminado. Errores: {len(errores)}")
    for e in errores:
        print(" -", e)