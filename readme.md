use coqui tts alongside svc models to get passable tts

todo:
try edge-tts for initial tts generation using their 'natural' voices

setup
```
python -m pip install -U pip setuptools wheel TTS
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U so-vits-svc-fork
```


