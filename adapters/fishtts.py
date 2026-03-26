"""Fish Speech adapter - Zero-shot voice cloning via native HTTP API.

Fish Speech: https://github.com/fishaudio/fish-speech
- ~5GB VRAM
- Fast inference (~200ms TTFB)
- Supports reference audio cloning and server-stored reference IDs
- Default port: 8080

This adapter targets the native Fish Speech API server (tools/api_server.py),
not the OpenAudio S1-Mini variant (which uses a different port/schema).

Voice selection (voice_path argument):
  - File path to a .wav/.mp3 file  → voice cloning via reference audio
  - A reference_id string          → server-stored voice (see /v1/references/list)
  - Empty string / None            → default model voice (no reference)
"""
import base64
import logging
import os
from pathlib import Path

import requests

from .base import TTSBackend

logger = logging.getLogger(__name__)


class FishTTSBackend(TTSBackend):
    """Native Fish Speech TTS backend.

    Connects to a running Fish Speech API server and supports:
    - Zero-shot voice cloning from a local reference audio file
    - Server-stored reference voices (reference_id)
    - Configurable sampling parameters (temperature, top_p, repetition_penalty)
    """

    def __init__(
        self,
        host: str = None,
        api_key: str = None,
        top_p: float = 0.8,
        temperature: float = 0.8,
        repetition_penalty: float = 1.1,
        chunk_length: int = 200,
    ):
        self.host = (
            host
            or os.environ.get("FISHTTS_HOST")
            or os.environ.get("FISHTTS_URL")
            or "http://localhost:8080"
        )
        self.api_key = api_key or os.environ.get("FISHTTS_API_KEY", "")
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.chunk_length = chunk_length

    @property
    def name(self) -> str:
        return "fishtts"

    @property
    def port(self) -> int:
        return 8080

    @property
    def vram_gb(self) -> int:
        return 5

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/v1/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        text: str,
        voice_path: str = "",
        transcript: str = "",
        **kwargs,
    ) -> bytes:
        """Generate speech using the native Fish Speech API.

        Args:
            text: Text to synthesize.
            voice_path: Either a path to a local reference audio file (for
                voice cloning) or a reference_id string stored on the server
                (e.g. "my-speaker"). Pass an empty string to use the default
                model voice.
            transcript: Transcript of the reference audio. Only used when
                voice_path is a file path. Improves cloning accuracy.
            **kwargs: Optional overrides for top_p, temperature,
                repetition_penalty, chunk_length, seed, normalize.

        Returns:
            WAV audio bytes.

        Raises:
            RuntimeError: If the server returns a non-200 response.
        """
        payload: dict = {
            "text": text,
            "format": "wav",
            "chunk_length": kwargs.get("chunk_length", self.chunk_length),
            "top_p": kwargs.get("top_p", self.top_p),
            "temperature": kwargs.get("temperature", self.temperature),
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
            "normalize": kwargs.get("normalize", True),
            "streaming": False,
        }

        if kwargs.get("seed") is not None:
            payload["seed"] = kwargs["seed"]

        if voice_path and Path(voice_path).is_file():
            # Local file → voice cloning via inline reference audio
            audio_bytes = Path(voice_path).read_bytes()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            payload["references"] = [{"audio": audio_b64, "text": transcript or ""}]
            payload["reference_id"] = None
        elif voice_path:
            # Non-path string → treat as a server-stored reference_id
            payload["references"] = []
            payload["reference_id"] = voice_path
        else:
            # No reference → default model voice
            payload["references"] = []
            payload["reference_id"] = None

        try:
            response = requests.post(
                f"{self.host}/v1/tts",
                json=payload,
                headers=self._headers(),
                timeout=120,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Fish Speech API error {exc.response.status_code}: {exc.response.text}"
            ) from exc

        return response.content

    def list_voices(self) -> list[str]:
        """List reference IDs stored on the Fish Speech server."""
        try:
            r = requests.get(
                f"{self.host}/v1/references/list",
                headers=self._headers(),
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("reference_ids", [])
        except Exception as exc:
            logger.warning(f"Failed to fetch reference IDs from Fish Speech: {exc}")
            return []
