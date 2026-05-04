"""
config.py — SpatialVoiceAI global configuration.

All runtime constants are sourced from environment variables with sane
defaults.  Import this module everywhere; never hard-code paths or magic
numbers elsewhere in the codebase.

Usage:
    from config import cfg
    print(cfg.SAMPLE_RATE)
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioConfig:
    SAMPLE_RATE: int            = field(default_factory=lambda: _env_int("AUDIO_SAMPLE_RATE", 16_000))
    CHANNELS: int               = field(default_factory=lambda: _env_int("AUDIO_CHANNELS", 1))
    # Frames per PyAudio callback — ~60ms at 16 kHz
    CHUNK_SIZE: int             = field(default_factory=lambda: _env_int("AUDIO_CHUNK_SIZE", 960))
    # Seconds of audio fed to Whisper per inference
    ASR_CHUNK_SECONDS: float    = field(default_factory=lambda: _env_float("ASR_CHUNK_SECONDS", 3.0))
    # Seconds of audio used for sliding-window diarization
    DIAR_WINDOW_SECONDS: float  = field(default_factory=lambda: _env_float("DIAR_WINDOW_SECONDS", 5.0))
    AUDIO_QUEUE_MAXSIZE: int    = field(default_factory=lambda: _env_int("AUDIO_QUEUE_MAXSIZE", 50))
    ASR_QUEUE_MAXSIZE: int      = field(default_factory=lambda: _env_int("ASR_QUEUE_MAXSIZE", 20))
    # PyAudio format constant (paInt16 = 8)
    PYAUDIO_FORMAT: int         = 8
    PYAUDIO_FRAMES_PER_BUFFER: int = field(default_factory=lambda: _env_int("AUDIO_CHUNK_SIZE", 960))


# ---------------------------------------------------------------------------
# Model Paths
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class ModelPaths:
    # Root for all downloaded model artefacts (gitignored)
    MODELS_DIR: Path = field(
        default_factory=lambda: Path(_env("MODELS_DIR", str(_BASE_DIR / "models")))
    )

    @property
    def WHISPER_MODEL(self) -> Path:
        name = _env("WHISPER_MODEL_FILE", "ggml-small.en.bin")
        return self.MODELS_DIR / "whisper" / name

    @property
    def PHI3_MODEL(self) -> Path:
        name = _env("PHI3_MODEL_FILE", "phi3-mini-4k-instruct-q4_k_m.gguf")
        return self.MODELS_DIR / "phi3" / name

    @property
    def HRTF_DIR(self) -> Path:
        return Path(_env("HRTF_DIR", str(_BASE_DIR / "hrtf" / "sadie2")))

    # sentence-transformers downloads this automatically; path is cache override
    MINILM_MODEL_NAME: str = "all-MiniLM-L6-v2"
    SPEECHBRAIN_MODEL: str = "speechbrain/spkrec-ecapa-voxceleb"


# ---------------------------------------------------------------------------
# LLM / Phi-3
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    # llama-cpp-python context window
    N_CTX: int          = field(default_factory=lambda: _env_int("PHI3_N_CTX", 4096))
    N_THREADS: int      = field(default_factory=lambda: _env_int("PHI3_N_THREADS", 4))
    N_GPU_LAYERS: int   = field(default_factory=lambda: _env_int("PHI3_N_GPU_LAYERS", 0))
    MAX_TOKENS: int     = field(default_factory=lambda: _env_int("PHI3_MAX_TOKENS", 512))
    TEMPERATURE: float  = field(default_factory=lambda: _env_float("PHI3_TEMPERATURE", 0.1))
    # Number of retries when JSON parsing fails
    JSON_RETRY_COUNT: int = 1


# ---------------------------------------------------------------------------
# Embeddings / FAISS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    EMBEDDING_DIM: int  = 384          # MiniLM output dimension
    FAISS_TOP_K: int    = field(default_factory=lambda: _env_int("FAISS_TOP_K", 5))
    # Minimum cosine similarity to create a semantic edge
    EDGE_SIM_THRESHOLD: float = field(
        default_factory=lambda: _env_float("EDGE_SIM_THRESHOLD", 0.45)
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

EventType = Literal["decision", "action_item", "question", "disagreement", "none"]

@dataclass(frozen=True)
class AgentConfig:
    # How many prior graph nodes are fed as context to EventDetectionAgent
    EVENT_CONTEXT_NODES: int    = field(default_factory=lambda: _env_int("EVENT_CONTEXT_NODES", 5))
    # How many FAISS results feed QAAgent
    QA_RETRIEVAL_TOP_K: int     = field(default_factory=lambda: _env_int("QA_RETRIEVAL_TOP_K", 5))
    VALID_EVENT_TYPES: tuple[str, ...] = (
        "decision", "action_item", "question", "disagreement", "none"
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DBConfig:
    DB_PATH: Path = field(
        default_factory=lambda: Path(_env("DB_PATH", str(_BASE_DIR / "spatialvoiceai.db")))
    )
    # Pre-seeded demo session path (fallback if live pipeline fails)
    DEMO_DB_PATH: Path = field(
        default_factory=lambda: Path(_env("DEMO_DB_PATH", str(_BASE_DIR.parent / "demo" / "seed.db")))
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    HOST: str   = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    PORT: int   = field(default_factory=lambda: _env_int("PORT", 8000))
    LOG_LEVEL: str = field(default_factory=lambda: _env("LOG_LEVEL", "info"))
    # CORS origins — comma-separated
    CORS_ORIGINS: list[str] = field(
        default_factory=lambda: _env("CORS_ORIGINS", "http://localhost:3000").split(",")
    )


# ---------------------------------------------------------------------------
# HRTF Spatial Positions
# ---------------------------------------------------------------------------

# Azimuth angles (degrees) for each speaker slot.
# Matched to Galaxy Buds Pro spatial audio azimuth range (330°/30°).
# SPK_0 = left-front, SPK_1 = right-front; extend for 3+ speakers.
SPEAKER_AZIMUTHS: dict[str, float] = {
    "SPK_0": -30.0,   # 330° (left)
    "SPK_1":  30.0,   # 030° (right)
    "SPK_2": -60.0,
    "SPK_3":  60.0,
}
SPEAKER_ELEVATION: float = 0.0   # degrees, ear-level


# ---------------------------------------------------------------------------
# Root config object — single import target
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    audio: AudioConfig      = field(default_factory=AudioConfig)
    models: ModelPaths      = field(default_factory=ModelPaths)
    llm: LLMConfig          = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    agent: AgentConfig      = field(default_factory=AgentConfig)
    db: DBConfig            = field(default_factory=DBConfig)
    server: ServerConfig    = field(default_factory=ServerConfig)


cfg = Config()


# ---------------------------------------------------------------------------
# Convenience aliases (backward compat + readability)
# ---------------------------------------------------------------------------

SAMPLE_RATE             = cfg.audio.SAMPLE_RATE
CHUNK_SIZE              = cfg.audio.CHUNK_SIZE
ASR_CHUNK_SECONDS       = cfg.audio.ASR_CHUNK_SECONDS
DIAR_WINDOW_SECONDS     = cfg.audio.DIAR_WINDOW_SECONDS
AUDIO_QUEUE_MAXSIZE     = cfg.audio.AUDIO_QUEUE_MAXSIZE
ASR_QUEUE_MAXSIZE       = cfg.audio.ASR_QUEUE_MAXSIZE

WHISPER_MODEL_PATH      = cfg.models.WHISPER_MODEL
PHI3_MODEL_PATH         = cfg.models.PHI3_MODEL
HRTF_DIR                = cfg.models.HRTF_DIR

DB_PATH                 = cfg.db.DB_PATH
DEMO_DB_PATH            = cfg.db.DEMO_DB_PATH
