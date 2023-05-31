import torch
import numpy as np
import contextlib
import logging
import librosa
from TTS.api import TTS
import sounddevice as sd

from so_vits_svc_fork.inference.core import Svc
from so_vits_svc_fork.utils import get_optimal_device

def play_audio(audio, sample_rate):
    sd.play(audio, sample_rate)
    sd.wait()

def text_to_speech(text, sr_target):
    # List available 🐸TTS models and choose the first one
    # print(TTS.list_models())
    model_name = TTS.list_models()[0]
    # model_name = 'tts_models/en/ljspeech/glow-tts'
    # Init TTS
    tts = TTS(model_name, gpu=True)
    # Text to speech to a file
    print(tts.speakers)
    print(tts.languages)
    tts.tts_to_file(text=text, speaker=tts.speakers[3], language=tts.languages[0], file_path="audio.wav", emotion="Angry")
    #multi speaker model requires these speaker=tts.speakers[0], language=tts.languages[0],
    #emotions
    #Neutral Dull Happy Sad Surprise Angry


    # Load audio with desired sample rate
    audio, _ = librosa.load('audio.wav', sr=sr_target)

    return audio

def load_model(model_path, config_path, cluster_model_path=""):
    print("Loading Model...")
    device: str | torch.device = get_optimal_device()

    svc_model = Svc(
            net_g_path=model_path,
            config_path=config_path,
            cluster_model_path=cluster_model_path
            if cluster_model_path
            else None,
            device=device,
        )

    print("Warming up the model...")
    svc_model.infer(
        speaker=0,
        transpose=0,
        auto_predict_f0=False,
        cluster_infer_ratio=0,
        noise_scale=0.4,
        f0_method="dio", #"crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
        audio=np.zeros(svc_model.target_sample, dtype=np.float32),
    )
    return svc_model

def so_vits_svc_tts(text, svc_model):
    with contextlib.redirect_stdout(None):
        audio = text_to_speech(text, svc_model.target_sample)

    audio = svc_model.infer_silence(
                    audio.astype(np.float32),
                    speaker=0,
                    transpose=0,
                    auto_predict_f0=True,
                    cluster_infer_ratio=0,
                    noise_scale=0.1,
                    f0_method="dio", #"crepe", "crepe-tiny", "parselmouth", "dio", "harvest"
                    db_thresh=-100,
                    pad_seconds=1,
                    chunk_seconds=0.5,
                    absolute_thresh=False,
                    max_chunk_seconds=40,
                )
    return audio

if __name__ == "__main__":
    # Set the logging level to a higher level
    logging.disable(logging.CRITICAL)

    model_path = ""
    config_path = ""
    model = load_model(model_path, config_path)

    while True:
        tts_in = input("Enter tts text: ")
        audio_out = so_vits_svc_tts(tts_in, model)
        play_audio(audio_out, model.target_sample)