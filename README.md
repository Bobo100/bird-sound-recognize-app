# 🐦 BirdSoundRecognizeAPP

> Android app that recognizes bird species from the microphone in real time, fully offline, using the ResNet model trained in [BirdSoundRecognize](https://github.com/Bobo100/BirdSoundRecognize).

<p align="center">
  <img src="image/resnet-flow%20chart.png" alt="ResNet real-time flow chart" width="720">
</p>

There is little material on how to turn live microphone audio into exactly the same input the model saw during training. This app does it on-device with [JLibrosa](https://github.com/Subtitle-Synchronizer/jlibrosa), reproducing the training-side `librosa` pipeline step by step. Because nothing is sent to a server, it saves at least 8 seconds per recognition compared with a server round trip.

## How it works

Every second of 44.1 kHz mono audio goes through the same steps as `utils/audio.melspec` in the training repo:

| Step | Training (Python, `librosa`) | App (`RealTimeRecognize3.kt`, JLibrosa) |
|---|---|---|
| Pre-emphasis | `y[n] = x[n] − 0.95·x[n−1]` | `FIR` with `[-0.95, 1.0]` |
| Mel spectrogram | `n_fft=1024`, `hop=172`, `n_mels=128`, 500–15000 Hz, magnitude, reflect padding | `generateMelSpectroGram(…, 1024, 128, 172)` |
| dB | `amplitude_to_db(ref=max, amin=1e-5, top_db=80)` | `AudioFeatureExtraction.amplitudeToDb` |
| Image | flip vertically, keep 128 × 256, min-max scale to 0–255 | `flip_vertically`, `reshape`, `normalize` |
| Model | `128 × 256 × 1` float input, pixel values 0–255 | `assets/resnet_model.tflite`, top-3 labels shown |

The app's features match the training pipeline to within ±1 grey level per pixel (verified against `librosa` on recordings from the training repo, see [`docs/plan/`](docs/plan/)).

## Project structure

```
app/src/main/
├── java/
│   ├── RealTime2/
│   │   ├── RealTimeRecognize3.kt     # Launcher activity: recording loop + feature pipeline
│   │   ├── AcousticEventDetector.kt  # Model / label file names
│   │   ├── Classifier.java           # TFLite interpreter wrapper, top-3 results
│   │   └── aed_method_DSP/FIR.kt     # Pre-emphasis filter
│   ├── JLibrosa/                     # Bundled JLibrosa, adjusted to match librosa
│   └── Variable/GlobalVariable.java  # Application class
├── assets/
│   ├── resnet_model.tflite           # 184-species ResNet
│   └── resnet_labels.txt
└── res/
docs/plan/                            # Records of why the project was changed
```

## Build

Open the project in Android Studio and run the `app` configuration on a device (minSdk 25). The app asks for microphone permission on first launch; press **Record** to start recognizing and **stop** to end.

To use your own model, replace `assets/resnet_model.tflite` and `assets/resnet_labels.txt` with the files `train.py` exports in the training repo.

## Status

Research code from a 2022 master's thesis, not actively maintained. The EfficientNet and YAMNet versions mentioned in the thesis are not included.

Planned when it was written: a fuller interface, a "results over the last 10 seconds" summary, and configurable window length instead of a fixed 1 second.

**Keywords:** Bird Sound, Bird Sound Recognition, Application, Machine Learning, Bird Identification
