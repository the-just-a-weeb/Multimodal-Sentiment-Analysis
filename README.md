# MMC Emotion Detector

## Requirements
### Hardware
- Webcam
- Microphone (not required for video-only mode)
- GPU recommended (CUDA or Apple MPS); CPU inference is supported but slower

### Python dependencies

Install into a virtual environment:

```bash
pip install torch torchvision torchaudio
pip install transformers timm einops nltk
pip install mediapipe opencv-python sounddevice websockets
pip install openai-whisper 
```

### Model files

The following files must be present in the `MMC_final/` directory before the server will start:

 `Video.pt`  Trained MAE-DFER video backbone (ViT-Base) 
 `audio_model.pt`  Trained HuBERT-based audio model 
 `best_model_v2.pt`  Trained DeBERTa-v3 + BiGRU text model 
 `fusion_best.pt`  Trained multi-modal fusion model 
`blaze_face_short_range.tflite`  MediaPipe BlazeFace detector 

### MAE-DFER source code

The video model depends on architecture code from the MAE-DFER repository. Clone it into the `content/` directory:

```bash
git clone https://github.com/sunlicai/MAE-DFER content/MAE-DFER
```

---

## Running the Server

Open a terminal, navigate to the `MMC_final/` directory, and choose a server variant:

### Full multi-modal (video + audio + text)
```bash
python mmc_server.py
```
Uses all three modalities. Requires all model files listed above plus a webcam and microphone. Whisper transcription is enabled automatically if `openai-whisper` is installed.

## Opening the Dashboard

After the server starts, open `mmc_dashboard.html` directly in any modern browser (Chrome, Firefox, Edge). The file runs as a local HTML page.

Click **Start Detector** to begin streaming.

---

## Dashboard Overview

The dashboard is a single-page application that connects to the server over WebSocket and visualizes emotion data in real time.

### Controls


**Start Detector**  begins capture and inference 
**Stop Detector**  halts capture and returns the server to idle 

### Connection behavior

The dashboard auto-connects to `ws://localhost:8765` on page load and reconnects automatically every 3 seconds if the connection drops. 

---

