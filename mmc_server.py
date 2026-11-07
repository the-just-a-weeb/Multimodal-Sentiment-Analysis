"""
mmc_server.py — Backend server for the MMC Emotion Detector dashboard.

Port 8765  WebSocket — all communication (commands in, data + frames out)

Client → Server:  {"cmd": "start"} | {"cmd": "stop"}
Server → Client:
  {"type": "status", "state": "idle|starting|running|stopping"}
  {"type": "data",   "emotion": str, "sentiment": float, "scores": {...},
                     "transcript": str, "audio_level": float, "spectrum": [...]}
  {"type": "frame",  "data": "<base64-jpeg>"}
"""

import asyncio
import base64
import json
import platform
import queue
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
import nltk
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import websockets
from einops import rearrange
from timm.models import create_model
from nltk.corpus import stopwords
from transformers import HubertModel, HubertConfig, AutoModel, AutoTokenizer

nltk.download('stopwords', quiet=True)

try:
    import whisper as _whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[WARNING] openai-whisper not installed — text modality disabled")

# ── Paths ──────────────────────────────────────────────────────────────────────

HERE        = Path(__file__).parent
MAE_DFER    = HERE / 'content/MAE-DFER'
VIDEO_CKPT  = HERE / 'Video.pt'
AUDIO_CKPT  = HERE / 'audio_model.pt'
TEXT_CKPT   = HERE / 'best_model_v2.pt'
FUSION_CKPT = HERE / 'fusion_best.pt'
TFLITE_PATH = str(HERE / 'blaze_face_short_range.tflite')
WHISPER_DIR = str(HERE / 'whisper_weights')

# ──────────────────────────────────────────────────────────────────────────────

if not MAE_DFER.exists():
    raise FileNotFoundError(
        f"MAE-DFER not found at {MAE_DFER}\n"
        "Run: git clone https://github.com/sunlicai/MAE-DFER content/MAE-DFER")

sys.path.insert(0, str(MAE_DFER))
import modeling_finetune

# ── Configuration ──────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

WIN_SEC             = 5
INFERENCE_INTERVAL  = 0.2
VIDEO_FPS           = 30
FACE_DET_THRESH     = 0.5
FACE_BBOX_PADDING   = 20
AUDIO_SAMPLE_RATE   = 16000
MIN_WIN_LEN_RATIO   = 0.9
MIN_WIN_SEC_OVERLAP = 0.8
SPECTRUM_BINS       = 64

NUM_FRAMES   = 16
HIDDEN_SIZE  = 512
NUM_TARGETS  = 7

EMOTIONS = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

# ── Consistency tuning ─────────────────────────────────────────────────────────
# Weight of acoustic heuristics blended into the fused model output.
# Lower than audio-only (0.50) because video+text already diversify predictions.
ACOUSTIC_BLEND    = 0.30
# Minimum blended score to display a named emotion; below → 'neutral'.
NEUTRAL_THRESHOLD = 0.25
# EMA weight for new prediction sample (higher = faster response, more jitter).
SMOOTH_ALPHA      = 0.35
# ─────────────────────────────────────────────────────────────────────────────

WS_PORT = 8765

# ── Shared state ───────────────────────────────────────────────────────────────

_det_state  = 'idle'
_emo_data   = {
    'emotion':     'neutral',
    'sentiment':   0.0,
    'scores':      {e: 0.0 for e in EMOTIONS},
    'transcript':  '',
    'audio_level': 0.0,
    'spectrum':    [0.0] * SPECTRUM_BINS,
}
_data_lock  = threading.Lock()
_ws_clients = set()
_ws_loop    = None
_manager    = None

# ── WebSocket broadcast ────────────────────────────────────────────────────────

def _publish(emotion: str, sentiment: float, scores: dict,
             transcript: str = '', spectrum=None, audio_level: float = 0.0):
    if spectrum is None:
        spectrum = [0.0] * SPECTRUM_BINS
    with _data_lock:
        _emo_data.update({
            'emotion':     emotion,
            'sentiment':   round(float(sentiment), 3),
            'scores':      {e: round(min(max(float(scores.get(e, 0.0)), 0.0), 1.0), 3) for e in EMOTIONS},
            'transcript':  transcript,
            'audio_level': round(float(audio_level), 3),
            'spectrum':    [round(float(v), 3) for v in spectrum],
        })
    if _ws_loop and _ws_loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast(), _ws_loop)


async def _send_all(msg: str):
    dead = set()
    for client in _ws_clients.copy():
        try:
            await client.send(msg)
        except Exception:
            dead.add(client)
    _ws_clients.difference_update(dead)


async def _broadcast():
    with _data_lock:
        d = dict(_emo_data)
    d['type'] = 'data'
    await _send_all(json.dumps(d))


async def _broadcast_status():
    await _send_all(json.dumps({'type': 'status', 'state': _det_state}))


def _push_status():
    if _ws_loop and _ws_loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast_status(), _ws_loop)


def _push_frame(frame: np.ndarray):
    if not _ws_clients or not (_ws_loop and _ws_loop.is_running()):
        return
    ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return
    b64 = base64.b64encode(jpg.tobytes()).decode('ascii')
    asyncio.run_coroutine_threadsafe(
        _send_all(json.dumps({'type': 'frame', 'data': b64})), _ws_loop)


async def _ws_handler(websocket):
    global _det_state
    _ws_clients.add(websocket)
    try:
        await websocket.send(json.dumps({'type': 'status', 'state': _det_state}))
        with _data_lock:
            d = dict(_emo_data)
        d['type'] = 'data'
        await websocket.send(json.dumps(d))

        async for raw in websocket:
            try:
                msg = json.loads(raw)
                cmd = msg.get('cmd')
                if cmd == 'start' and _det_state == 'idle':
                    _det_state = 'starting'
                    await _broadcast_status()
                    threading.Thread(target=_start_detector, daemon=True).start()
                elif cmd == 'stop' and _det_state == 'running':
                    _det_state = 'stopping'
                    await _broadcast_status()
                    threading.Thread(target=_stop_detector, daemon=True).start()
            except Exception as exc:
                print(f'[WS cmd] {exc}')
    finally:
        _ws_clients.discard(websocket)


def _run_ws():
    global _ws_loop
    _ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ws_loop)

    async def _serve():
        async with websockets.serve(
            _ws_handler, '0.0.0.0', WS_PORT,
            ping_interval=None,   # browser handles keepalive; avoids spurious drops
        ):
            await asyncio.Future()

    _ws_loop.run_until_complete(_serve())


# ── MAE-DFER LGBlock patch ─────────────────────────────────────────────────────

def _patch_lgblock():
    def _forward(self, x, b):
        n = x.shape[0] // b
        if self.gamma_1 is None:
            if self.first_attn_type == 'self':
                x = x + self.drop_path(self.first_attn(self.first_attn_norm0(x)))
            else:
                x[:, :1] = x[:, :1] + self.drop_path(
                    self.first_attn(self.first_attn_norm0(x[:, :1]),
                                    context=self.first_attn_norm1(x[:, 1:])))
            if not self.no_second:
                mt = rearrange(x[:, 0].clone(), '(b n) c -> b n c', b=b)
                mt = mt + self.drop_path(self.second_attn(self.second_attn_norm0(mt)))
                x[:, 0] = rearrange(mt, 'b n c -> (b n) c')
            else:
                mt = rearrange(x[:, 0].clone(), '(b n) c -> b n c', b=b)
            if not self.no_third:
                if self.third_attn_type == 'self':
                    x = x + self.drop_path(self.third_attn(self.third_attn_norm0(x)))
                else:
                    lt = rearrange(x[:, 1:].clone(), '(b n) s c -> b (n s) c', b=b)
                    lt = lt + self.drop_path(
                        self.third_attn(self.third_attn_norm0(lt),
                                        context=self.third_attn_norm1(mt)))
                    x[:, 1:] = rearrange(lt, 'b (n s) c -> (b n) s c', n=n)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.first_attn_type == 'self':
                x = x + self.drop_path(self.gamma_0 * self.first_attn(self.first_attn_norm0(x)))
            else:
                x[:, :1] = x[:, :1] + self.drop_path(
                    self.gamma_0 * self.first_attn(self.first_attn_norm0(x[:, :1]),
                                                   context=self.first_attn_norm1(x[:, 1:])))
            if not self.no_second:
                mt = rearrange(x[:, 0].clone(), '(b n) c -> b n c', b=b)
                mt = mt + self.drop_path(self.gamma_1 * self.second_attn(self.second_attn_norm0(mt)))
                x[:, 0] = rearrange(mt, 'b n c -> (b n) c')
            else:
                mt = rearrange(x[:, 0].clone(), '(b n) c -> b n c', b=b)
            if not self.no_third:
                if self.third_attn_type == 'self':
                    x = x + self.drop_path(self.gamma_1 * self.third_attn(self.third_attn_norm0(x)))
                else:
                    lt = rearrange(x[:, 1:].clone(), '(b n) s c -> b (n s) c', b=b)
                    lt = lt + self.drop_path(
                        self.gamma_1 * self.third_attn(self.third_attn_norm0(lt),
                                                       context=self.third_attn_norm1(mt)))
                    x[:, 1:] = rearrange(lt, 'b (n s) c -> (b n) s c', n=n)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    modeling_finetune.LGBlock.forward = _forward


# ── Video model (MAE-DFER) ────────────────────────────────────────────────────

class MAEDFERForEmotionRegression(nn.Module):
    def __init__(self, num_targets=7, dropout=0.3):
        super().__init__()
        self.backbone = create_model(
            'vit_base_dim512_no_depth_patch16_160',
            pretrained=False, num_classes=7,
            all_frames=NUM_FRAMES, tubelet_size=2,
            use_mean_pooling=True, attn_type='local_global',
            lg_region_size=(2, 5, 10),
            lg_first_attn_type='self',    lg_third_attn_type='cross',
            lg_attn_param_sharing_first_third=False,
            lg_attn_param_sharing_all=False,
            lg_classify_token_type='org', lg_no_second=False, lg_no_third=False,
            drop_path_rate=0.1, init_values=0.1,
        )
        self.backbone.head = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_SIZE, 256), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_targets),
        )

    def get_features(self, x):
        return self.backbone(x)  # (B, 512) mean-pooled backbone output

    def forward(self, x):
        f  = self.backbone(x)
        lg = self.head(f)
        emotions  = torch.clamp(torch.relu(lg[:, :6]), 0.0, 3.0)
        sentiment = torch.tanh(lg[:, 6:7]) * 3.0
        return torch.cat([emotions, sentiment], dim=1)


# ── Audio model (HuBERT-base backbone) ───────────────────────────────────────

class AudioEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hubert = HubertModel(HubertConfig())
        self.attn_pool = nn.Linear(768, 1)
        self.emotion_head = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 6))
        self.sentiment_head = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def get_features(self, input_values):
        # input_values: (B, T_samples) raw waveform at 16 kHz
        return self.hubert(
            input_values=input_values, return_dict=True
        ).last_hidden_state  # (B, T_frames, 768)


# ── Text model (DeBERTa-v3-base + BiGRU) ─────────────────────────────────────

class TextEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('microsoft/deberta-v3-base')
        self.gru = nn.GRU(768, 256, batch_first=True, bidirectional=True)
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 6))
        self.sentiment_head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def get_features(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state


# ── Fusion model building blocks ─────────────────────────────────────────────

class _CrossAttnLayer(nn.Module):
    """Pre-norm cross-attention layer: query from one modality, key/value from another."""
    def __init__(self, dim=256, num_heads=8, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, q, kv):
        kv_n = self.norm_kv(kv)
        q = q + self.attn(self.norm_q(q), kv_n, kv_n)[0]
        q = q + self.ff(self.norm_ff(q))
        return q


class _CrossModal(nn.Module):
    """Stack of cross-attention layers: one modality attends to another."""
    def __init__(self, dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [_CrossAttnLayer(dim, num_heads) for _ in range(num_layers)])

    def forward(self, q, kv):
        for layer in self.layers:
            q = layer(q, kv)
        return q


# ── Middle-fusion model ───────────────────────────────────────────────────────

class MiddleFusionModel(nn.Module):
    """
    Cross-modal pairwise fusion model (fusion_best.pt architecture).

    Data flow:
      1. Conv1d project each modality: 768/512 → 256-d + learned positional encoding
      2. Pairwise cross-attention between all modality pairs (6 modules, 256-d)
      3. Concatenate each modality's two cross-attended outputs → 512-d sequence
      4. Self-attention on concatenated sequences (512-d, 2 layers per modality)
      5. Mean-pool → concatenate three 512-d vectors → 1536-d
      6. LayerNorm → emotion head (6) + sentiment head (1)
    """
    PROJ_DIM = 256

    def __init__(self):
        super().__init__()
        D = self.PROJ_DIM

        # Learned positional encodings (max sequence lengths)
        self.pos_t = nn.Parameter(torch.zeros(1, 64,  D))
        self.pos_a = nn.Parameter(torch.zeros(1, 128, D))
        self.pos_v = nn.Parameter(torch.zeros(1, 32,  D))

        # Conv1d projections (bias=False, padding=1 for same-length output)
        self.proj_t = nn.Conv1d(768, D, kernel_size=3, padding=1, bias=False)
        self.proj_a = nn.Conv1d(768, D, kernel_size=3, padding=1, bias=False)
        self.proj_v = nn.Conv1d(512, D, kernel_size=3, padding=1, bias=False)

        # Pairwise cross-attention (all at 256-d)
        self.t_from_a = _CrossModal(D)
        self.t_from_v = _CrossModal(D)
        self.a_from_t = _CrossModal(D)
        self.a_from_v = _CrossModal(D)
        self.v_from_t = _CrossModal(D)
        self.v_from_a = _CrossModal(D)

        # Self-attention on concatenated cross-attended features (512-d = 2×256)
        self.self_t = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024,
                                       batch_first=True), num_layers=2)
        self.self_a = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024,
                                       batch_first=True), num_layers=2)
        self.self_v = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024,
                                       batch_first=True), num_layers=2)

        # Fusion LayerNorm: 1536 = 3 modalities × 512
        self.fusion_norm = nn.LayerNorm(1536)

        # Output heads (indices 0 and 3 match checkpoint naming through Sequential)
        self.emo_head = nn.Sequential(
            nn.Linear(1536, D), nn.GELU(), nn.Dropout(0.1), nn.Linear(D, 6))
        self.sent_head = nn.Sequential(
            nn.Linear(1536, D), nn.GELU(), nn.Dropout(0.1), nn.Linear(D, 1))

    def _proj_pos(self, x, proj, pos):
        """(B, T, D_in) → Conv1d → (B, T, 256) + positional encoding."""
        T = x.size(1)
        z = proj(x.transpose(1, 2)).transpose(1, 2)
        return z + pos[:, :T, :]

    def forward(self, text_feat, audio_feat, video_feat):
        # Project + positional encoding
        t = self._proj_pos(text_feat,  self.proj_t, self.pos_t)   # (B, T_t, 256)
        a = self._proj_pos(audio_feat, self.proj_a, self.pos_a)   # (B, T_a, 256)
        v = self._proj_pos(video_feat, self.proj_v, self.pos_v)   # (B, T_v, 256)

        # Pairwise cross-attention
        tfa = self.t_from_a(t, a)
        tfv = self.t_from_v(t, v)
        aft = self.a_from_t(a, t)
        afv = self.a_from_v(a, v)
        vft = self.v_from_t(v, t)
        vfa = self.v_from_a(v, a)

        # Cat pairs → self-attention → mean pool
        t_fused = self.self_t(torch.cat([tfa, tfv], dim=-1)).mean(1)   # (B, 512)
        a_fused = self.self_a(torch.cat([aft, afv], dim=-1)).mean(1)
        v_fused = self.self_v(torch.cat([vft, vfa], dim=-1)).mean(1)

        fused = self.fusion_norm(
            torch.cat([t_fused, a_fused, v_fused], dim=-1))   # (B, 1536)

        emotions  = torch.clamp(torch.relu(self.emo_head(fused)), 0.0, 3.0)
        sentiment = torch.tanh(self.sent_head(fused)) * 3.0
        return torch.cat([emotions, sentiment], dim=-1)   # (B, 7)


# ── Multi-modal emotion detector ──────────────────────────────────────────────

class MMEmotionDetector:
    def __init__(self):
        self.detected_emotion = 'neutral'
        self.sentiment        = 0.0
        self._tokenizer       = None
        self._smooth          = {e: 0.0 for e in EMOTIONS}

        # ── Video (MAE-DFER) ──────────────────────────────────────────────────
        print("[MMC] Loading video model…")
        self.video_model = MAEDFERForEmotionRegression(num_targets=NUM_TARGETS)
        self.video_model.load_state_dict(
            torch.load(VIDEO_CKPT, map_location=DEVICE, weights_only=False))
        self.video_model.to(DEVICE).eval()
        print("[MMC] Video model loaded ✓")

        # ── Audio (HuBERT) ────────────────────────────────────────────────────
        print("[MMC] Loading audio model (HuBERT-base)…")
        self.audio_model = AudioEmotionModel()
        self.audio_model.load_state_dict(
            torch.load(AUDIO_CKPT, map_location=DEVICE, weights_only=False))
        self.audio_model.to(DEVICE).eval()
        print("[MMC] Audio model loaded ✓")

        # ── Text (DeBERTa-v3 + BiGRU) ─────────────────────────────────────────
        print("[MMC] Loading text model (DeBERTa-v3-base)…")
        self.text_model = self._init_text_model()

        # ── Fusion ────────────────────────────────────────────────────────────
        print("[MMC] Loading fusion model…")
        self.fusion_model = MiddleFusionModel()
        ckpt = torch.load(FUSION_CKPT, map_location=DEVICE, weights_only=False)
        self.fusion_model.load_state_dict(ckpt['model'])
        self.fusion_model.to(DEVICE).eval()
        print("[MMC] Fusion model loaded ✓")
        print("[MMC] All models ready.")

    # ── Text model loader ─────────────────────────────────────────────────────

    def _init_text_model(self):
        # Apply numpy 2.x compat patch here only — applying it at module level
        # corrupts C-extension init for cv2/mediapipe/torch loaded together.
        import sys as _s, numpy as _n
        _s.modules.setdefault('numpy._core', _n.core)
        _s.modules.setdefault('numpy._core.multiarray', _n.core.multiarray)
        try:
            model = TextEmotionModel()
            ckpt  = torch.load(TEXT_CKPT, map_location=DEVICE, weights_only=False)
            missing, unexpected = model.load_state_dict(
                ckpt['model_state_dict'], strict=False)
            if missing:
                print(f"[MMC] Text: {len(missing)} missing key(s) — {missing[:2]}")
            if unexpected:
                print(f"[MMC] Text: {len(unexpected)} unexpected key(s) — {unexpected[:2]}")
            model.to(DEVICE).eval()
            self._tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
            print("[MMC] Text model loaded ✓")
            return model
        except Exception as exc:
            print(f"[MMC] Text model unavailable ({exc}) — zero text features used")
            return None

    # ── Feature helpers ───────────────────────────────────────────────────────

    def _text_features(self, text: str):
        """Returns DeBERTa last_hidden_state (1, T, 768) or zeros if unavailable."""
        if not self.text_model or not text or not text.strip():
            return torch.zeros(1, 1, 768, device=DEVICE)
        try:
            enc = self._tokenizer(
                text, return_tensors='pt', truncation=True,
                max_length=64, padding='max_length')
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                return self.text_model.get_features(
                    enc['input_ids'], enc['attention_mask'])
        except Exception as exc:
            print(f"[MMC] Text feature error: {exc}")
            return torch.zeros(1, 1, 768, device=DEVICE)

    # ── Acoustic feature heuristics ──────────────────────────────────────────

    def _acoustic_scores(self, audio: np.ndarray, volume: float) -> dict:
        """
        Derive per-emotion weights from raw acoustic properties.

        audio  : amplitude-normalised waveform (max ≈ 1.0).
        volume : RMS-based level [0, 1] from AudioRetriever.get_level().

        Feature mapping (v=volume, c=spectral centroid, z=ZCR):
          happiness  → bright, energetic, moderate-fast
          sadness    → quiet, dark, slow
          anger      → loud, harsh, mid-pitch
          fear       → quiet but tense / high-pitch
          disgust    → flat, low-pitch, moderate
          surprise   → loud + sudden brightness
        """
        win      = np.hanning(len(audio))
        fft_mag  = np.abs(np.fft.rfft(audio * win))
        freqs    = np.fft.rfftfreq(len(audio), d=1.0 / AUDIO_SAMPLE_RATE)
        centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-8))
        c = float(np.clip((centroid - 100.0) / 3900.0, 0.0, 1.0))  # 100–4000 Hz → 0..1

        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)
        z   = float(min(zcr * 25.0, 1.0))

        v = float(np.clip(volume, 0.0, 1.0))

        return {
            'happiness': 0.35 * c + 0.30 * z + 0.35 * v,
            'sadness':   0.35 * (1 - v) + 0.35 * (1 - c) + 0.30 * (1 - z),
            'anger':     0.40 * v + 0.35 * z + 0.25 * (1 - c),
            'fear':      0.35 * c + 0.35 * (1 - v) + 0.30 * z,
            'disgust':   0.40 * (1 - c) + 0.30 * z + 0.30 * (1 - v),
            'surprise':  0.40 * v + 0.35 * c + 0.25 * z,
        }

    # ── Main predict ──────────────────────────────────────────────────────────

    def predict(self, video_tensor, audio_array, text_input=None, volume: float = 0.5):
        """
        video_tensor : (1, 3, 16, 160, 160) float32 tensor
        audio_array  : (T_samples,) amplitude-normalised float32 at 16 kHz
        text_input   : transcript string or None
        volume       : RMS-based audio level [0, 1] from AudioRetriever.get_level()
        Returns (dominant_emotion, sentiment, scores_dict)
        """
        with torch.no_grad():
            # Video backbone → (1, 512) → (1, 1, 512) single-frame sequence for fusion
            v_feat = self.video_model.get_features(
                video_tensor.to(DEVICE)).unsqueeze(1)

            # Audio HuBERT → (1, T_audio, 768); subsample to ≤128 frames (pos enc limit)
            audio_t = torch.from_numpy(audio_array).unsqueeze(0).to(DEVICE)
            a_feat  = self.audio_model.get_features(audio_t)
            if a_feat.size(1) > 128:
                idx    = torch.linspace(0, a_feat.size(1) - 1, 128).long()
                a_feat = a_feat[:, idx, :]

            # Text DeBERTa → (1, T_text, 768) or zeros
            t_feat = self._text_features(text_input or '')

            # Fusion → (1, 7)
            out = self.fusion_model(t_feat, a_feat, v_feat)

        arr          = out.squeeze().cpu().numpy()
        model_scores = {EMOTIONS[i]: float(np.clip(arr[i] / 3.0, 0.0, 1.0))
                        for i in range(6)}
        sentiment    = float(arr[6])

        # Blend fused model output with acoustic heuristics for diversity.
        acoustic = self._acoustic_scores(audio_array, volume)
        blended  = {
            e: (1.0 - ACOUSTIC_BLEND) * model_scores[e] + ACOUSTIC_BLEND * acoustic[e]
            for e in EMOTIONS
        }

        # EMA smoothing — prevents rapid label flipping between inference cycles.
        for e in EMOTIONS:
            self._smooth[e] = (1.0 - SMOOTH_ALPHA) * self._smooth[e] + SMOOTH_ALPHA * blended[e]

        dominant = max(self._smooth, key=self._smooth.get)
        if self._smooth[dominant] < NEUTRAL_THRESHOLD:
            dominant = 'neutral'

        self.detected_emotion = dominant
        self.sentiment        = sentiment
        return dominant, sentiment, dict(self._smooth)

    def get_detected_emotion(self):   return self.detected_emotion
    def get_detected_sentiment(self): return self.sentiment


# ── Retrievers ─────────────────────────────────────────────────────────────────

class AudioRetriever:
    def __init__(self):
        self.max_len = int(AUDIO_SAMPLE_RATE * WIN_SEC)
        self.min_len = int(self.max_len * MIN_WIN_LEN_RATIO)
        self._buf    = deque(maxlen=self.max_len)
        self._lock   = threading.Lock()
        self._stream = None
        self._active = False
        try:
            info = sd.query_devices(kind='input')
            print(f"[Audio] Default input device: {info['name']}")
        except Exception as exc:
            raise RuntimeError(
                f"No audio input device found: {exc}\n"
                "On macOS grant Microphone permission to Terminal in "
                "System Preferences → Security & Privacy → Privacy → Microphone."
            ) from exc

    def start_retrieving(self):
        self._active = True
        self._stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype='float32',
            blocksize=512, device=None, callback=self._cb)
        self._stream.start()

    def _cb(self, indata, frames, time_info, status):
        if not self._active:
            return
        t = time.time()
        with self._lock:
            for s in indata[:, 0]:
                self._buf.append((s, t))

    def stop_retrieving(self):
        self._active = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def get_window(self):
        with self._lock:
            buf = list(self._buf)
        if len(buf) < self.min_len:
            return None
        audio = np.array([s for s, _ in buf], dtype=np.float32)
        audio /= (np.max(np.abs(audio)) + 1e-8)
        return audio, (buf[0][1], buf[-1][1])

    def get_level(self):
        with self._lock:
            if not self._buf:
                return 0.0
            samples = np.array([s for s, _ in self._buf], dtype=np.float32)
        return float(min(np.sqrt(np.mean(samples ** 2)) * 10.0, 1.0))

    def get_spectrum(self, n_bins: int = SPECTRUM_BINS):
        with self._lock:
            if len(self._buf) < 512:
                return [0.0] * n_bins
            samples = np.array([s for s, _ in self._buf], dtype=np.float32)
        win   = np.hanning(len(samples))
        mag   = np.abs(np.fft.rfft(samples * win))
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / AUDIO_SAMPLE_RATE)
        lo, hi = 20.0, AUDIO_SAMPLE_RATE / 2.0
        edges  = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
        bins = []
        for i in range(n_bins):
            mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
            bins.append(float(np.mean(mag[mask])) if mask.any() else 0.0)
        max_val = max(bins) if max(bins) > 1e-8 else 1.0
        return [b / max_val for b in bins]


class VideoRetriever:
    def __init__(self):
        self.max_len = int(WIN_SEC * VIDEO_FPS)
        self.min_len = int(self.max_len * MIN_WIN_LEN_RATIO)
        self._buf    = deque(maxlen=self.max_len)
        self._lock   = threading.Lock()
        self._thread = None
        if platform.system() == 'Darwin':
            self._cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Cannot open camera (device 0).\n"
                "On macOS grant Camera permission to Terminal in "
                "System Preferences → Security & Privacy → Privacy → Camera."
            )
        self._cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Video] Camera opened: {w}x{h} @ {VIDEO_FPS} fps")
        self._disp   = None
        self._bbox   = None
        self._active = False

        base = mp.tasks.BaseOptions(model_asset_path=TFLITE_PATH)
        opts = mp.tasks.vision.FaceDetectorOptions(
            base_options=base, min_detection_confidence=FACE_DET_THRESH)
        self._det = mp.tasks.vision.FaceDetector.create_from_options(opts)

    def start_retrieving(self):
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        consecutive_fail = 0
        while self._active:
            ret, frame = self._cap.read()
            if not ret:
                consecutive_fail += 1
                if consecutive_fail >= 30:
                    print("[VideoRetriever] Camera stopped delivering frames — giving up.")
                    break
                time.sleep(0.1)
                continue
            consecutive_fail = 0
            self._disp = frame.copy()
            cropped    = self._crop_face(frame)
            with self._lock:
                self._buf.append((cropped, time.time()))

    def stop_retrieving(self):
        self._active = False
        if self._thread:
            self._thread.join(timeout=3)
        self._cap.release()

    def get_disp_frame(self): return self._disp
    def get_face_bbox(self):  return self._bbox

    def get_window(self):
        with self._lock:
            buf = list(self._buf)
        if len(buf) < self.min_len:
            return None
        frames = [f for f, _ in buf]
        return self._preprocess(frames), (buf[0][1], buf[-1][1])

    def _crop_face(self, frame):
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res  = self._det.detect(img)
        if res.detections:
            bb = res.detections[0].bounding_box
            x1 = max(0, bb.origin_x - FACE_BBOX_PADDING)
            y1 = max(0, bb.origin_y - FACE_BBOX_PADDING)
            x2 = min(w, bb.origin_x + bb.width  + FACE_BBOX_PADDING)
            y2 = min(h, bb.origin_y + bb.height + FACE_BBOX_PADDING)
        else:
            s  = min(h, w) // 2
            cx, cy = w // 2, h // 2
            x1, y1, x2, y2 = cx - s, cy - s, cx + s, cy + s
        self._bbox = (x1, y1, x2, y2)
        return cv2.resize(frame[y1:y2, x1:x2], (160, 160))

    def _preprocess(self, frames):
        out = []
        for f in frames:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            f = (f - 0.5) / 0.5
            out.append(f)
        idx = np.linspace(0, len(out) - 1, NUM_FRAMES, dtype=int)
        out = np.stack([out[i] for i in idx], axis=0)
        return torch.from_numpy(out).permute(3, 0, 1, 2).unsqueeze(0)


class TextRetriever:
    def __init__(self, model_size='tiny', language=None):
        if not WHISPER_AVAILABLE:
            self.model = None
        else:
            import os
            os.makedirs(WHISPER_DIR, exist_ok=True)
            self.model = _whisper.load_model(model_size, download_root=WHISPER_DIR)
        self.language    = language
        self._q          = queue.Queue()
        self._lock       = threading.Lock()
        self._transcript = ''
        self._time_range = None
        self._running    = False

    def start(self):
        if self.model is None:
            return
        self._running = True
        threading.Thread(target=self._worker, daemon=True, name='Transcriber').start()

    def stop(self):
        self._running = False

    _WHISPER_SEC = 6

    def put_window(self, audio: np.ndarray, time_range):
        if self.model is not None:
            self._q.put((audio, time_range))

    def get_transcript(self) -> str:
        with self._lock:
            return self._transcript

    def clean(self, text):
        text = text.lower()
        text = re.sub('[^a-z A-Z 0-9-]+', '', text)
        text = " ".join([w for w in text.split() if w not in stopwords.words('english')])
        return text

    def _worker(self):
        acc      = np.zeros(0, dtype=np.float32)
        acc_time = None
        max_samp = int(AUDIO_SAMPLE_RATE * self._WHISPER_SEC)
        min_samp = int(AUDIO_SAMPLE_RATE * 3)

        print('[Whisper] worker thread started')
        while self._running:
            try:
                audio, time_range = self._q.get(timeout=2.0)
            except queue.Empty:
                continue

            acc      = np.concatenate([acc, audio])
            acc_time = time_range
            if len(acc) > max_samp:
                acc = acc[-max_samp:]
            if len(acc) < min_samp:
                continue

            try:
                chunk  = acc / (np.max(np.abs(acc)) + 1e-8)
                result = self.model.transcribe(
                    chunk, beam_size=1,
                    language=self.language or 'en',
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0,
                )
                segments = result.get('segments', [])
                parts = []
                for seg in segments:
                    cleaned = self.clean(seg["text"].strip())
                    if cleaned:
                        parts.append(cleaned)
                if parts:
                    with self._lock:
                        self._transcript = ' '.join(parts)
                        self._time_range = acc_time
            except Exception as exc:
                print(f'[Transcriber] {exc}')


# ── Detector lifecycle manager ─────────────────────────────────────────────────

class DetectorManager:
    def __init__(self):
        self._video = None
        self._audio = None
        self._text  = None
        self._det   = None
        self._alive = False

    def start(self):
        global _det_state
        try:
            self._det   = MMEmotionDetector()
            self._video = VideoRetriever()
            self._audio = AudioRetriever()
            self._text  = TextRetriever()
            self._video.start_retrieving()
            self._audio.start_retrieving()
            self._text.start()
            self._alive = True
            _det_state  = 'running'
            _push_status()
            threading.Thread(target=self._infer_loop, daemon=True).start()
            threading.Thread(target=self._frame_loop, daemon=True).start()
            threading.Thread(target=self._audio_loop, daemon=True).start()
            print("Detector started.")
        except Exception as exc:
            print(f"[ERROR] Detector failed to start: {exc}")
            import traceback; traceback.print_exc()
            _det_state = 'idle'
            _push_status()

    def stop(self):
        global _det_state
        self._alive = False
        if self._video: self._video.stop_retrieving()
        if self._audio: self._audio.stop_retrieving()
        if self._text:  self._text.stop()
        _det_state = 'idle'
        _push_status()
        print("Detector stopped.")

    def _infer_loop(self):
        while self._alive:
            vw = self._video.get_window()
            aw = self._audio.get_window()
            if vw is None or aw is None:
                time.sleep(0.1)
                continue
            v_frames, v_range = vw
            a_audio,  a_range = aw
            if not _is_synced([v_range, a_range]):
                time.sleep(0.1)
                continue

            transcript  = self._text.get_transcript()
            spectrum    = self._audio.get_spectrum()
            audio_level = self._audio.get_level()

            dominant, sentiment, scores = self._det.predict(
                v_frames, a_audio, transcript, audio_level)
            _publish(dominant, sentiment, scores, transcript, spectrum, audio_level)

            time.sleep(INFERENCE_INTERVAL)

    def _audio_loop(self):
        last_text_feed = 0.0
        while self._alive:
            try:
                level    = self._audio.get_level()
                spectrum = self._audio.get_spectrum()

                if _ws_loop and _ws_loop.is_running() and _ws_clients:
                    asyncio.run_coroutine_threadsafe(
                        _send_all(json.dumps({
                            'type':        'audio',
                            'audio_level': round(float(level), 3),
                            'spectrum':    [round(float(v), 3) for v in spectrum],
                            'transcript':  self._text.get_transcript(),
                        })),
                        _ws_loop,
                    )

                now = time.time()
                if now - last_text_feed >= WIN_SEC:
                    raw = np.array([s for s, _ in list(self._audio._buf)],
                                   dtype=np.float32)
                    if len(raw) > 0:
                        self._text.put_window(raw, (now - WIN_SEC, now))
                        last_text_feed = now

            except Exception as exc:
                print(f'[AudioLoop] {exc}')
            time.sleep(0.1)

    def _frame_loop(self):
        font       = cv2.FONT_HERSHEY_SIMPLEX
        last_push  = 0.0
        PUSH_EVERY = 1.0 / 15
        while self._alive:
            frame = self._video.get_disp_frame()
            if frame is not None:
                frame = frame.copy()
                bbox  = self._video.get_face_bbox()
                if bbox:
                    x1, y1, x2, y2 = bbox
                    emo  = self._det.get_detected_emotion()
                    sent = self._det.get_detected_sentiment()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
                    label = f"{emo}  {sent:+.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, font, 0.7, 2)
                    ly = y1 - 10 if y1 - lh - 10 >= 0 else y2 + lh + 10
                    cv2.putText(frame, label, (x1, ly), font, 0.7, (0, 255, 80), 2)
                now = time.time()
                if now - last_push >= PUSH_EVERY:
                    _push_frame(frame)
                    last_push = now
            time.sleep(1 / 30)


def _is_synced(ranges):
    overlap = min(r[1] for r in ranges) - max(r[0] for r in ranges)
    return overlap >= WIN_SEC * MIN_WIN_SEC_OVERLAP


def _start_detector():
    global _manager
    _manager = DetectorManager()
    _manager.start()


def _stop_detector():
    global _manager
    if _manager:
        _manager.stop()
        _manager = None


# ── Entry point ────────────────────────────────────────────────────────────────

def _print_devices():
    import os

    print("-- Audio input devices ------------------------------------------")
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            marker = '>' if i == sd.default.device[0] else ' '
            print(f"  {marker} [{i}] {d['name']}")

    print("-- Video cameras ------------------------------------------------")
    backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else cv2.CAP_ANY
    null_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(null_fd, 2)
    try:
        for idx in range(8):
            cap = cv2.VideoCapture(idx, backend)
            opened = cap.isOpened()
            if opened:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if not opened:
                break
            marker = '>' if idx == 0 else ' '
            os.dup2(saved_fd, 2)
            print(f"  {marker} [{idx}] {w}x{h}")
            os.dup2(null_fd, 2)
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(null_fd)

    print("-" * 57)


def run():
    _patch_lgblock()
    _print_devices()

    threading.Thread(target=_run_ws, daemon=True).start()
    time.sleep(0.3)

    print()
    print("MMC Server ready.")
    print(f"  WS  →  ws://localhost:{WS_PORT}")
    print()
    print("Open  mmc_dashboard.html  in your browser, then click  Start Detector.")
    print("Press Ctrl+C to quit.")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down…")
        if _manager:
            _manager.stop()


if __name__ == '__main__':
    run()
