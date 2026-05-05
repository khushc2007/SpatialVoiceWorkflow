"""
backend/audio/hrtf_renderer.py
-------------------------------
Binaural HRTF renderer using SADIE II impulse responses.

APPROACH — fftconvolve stereo rendering
----------------------------------------
An HRTF (Head-Related Transfer Function) describes how a sound from a specific
azimuth/elevation is modified by the shape of the human head and ears before
reaching the eardrums.

The SADIE II dataset provides measured impulse responses (IRs) for Subject 002
at many azimuth angles. Each IR is a pair of short WAV arrays:
  - left_IR  : what the LEFT ear hears for a sound at angle θ
  - right_IR : what the RIGHT ear hears for a sound at angle θ

To spatialise a mono audio chunk at angle θ:
  1. Load left_IR and right_IR for the nearest measured azimuth to θ
  2. Convolve chunk with left_IR  → left_channel
  3. Convolve chunk with right_IR → right_channel
  4. Stack → stereo output array of shape (N, 2)

We use scipy.signal.fftconvolve (FFT-based) instead of np.convolve because:
  - chunk lengths are ~960 samples (20ms at 48kHz)
  - IR lengths are ~256 samples
  - fftconvolve is O(N log N) vs O(N·M) for direct convolution
  - Measured on i5-8th gen: <10ms per chunk ✓

SADIE II file layout assumed:
  {hrtf_dir}/
    Subject_002/
      HRIR_44100/
        azi_{angle}_ele_0.wav   # stereo WAV, ch0=left IR, ch1=right IR

TEAMMATE: Implement the three TODO sections.
  - __init__: load all IRs from disk, build azimuth→IR lookup dict
  - _nearest_azimuth: find closest measured angle to requested azimuth
  - render: fftconvolve and return stereo np.ndarray
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Speaker azimuth positions — must match frontend SpatialCanvas.tsx orb positions
SPEAKER_AZIMUTHS: Dict[str, float] = {
    "SPK_0": 330.0,   # left  (330° = -30° in standard convention)
    "SPK_1":  30.0,   # right
}


class HRTFRenderer:
    """Real-time binaural renderer using SADIE II Head-Related Impulse Responses.

    Parameters
    ----------
    hrtf_dir : str | Path
        Root directory of the SADIE II dataset.
        Expected layout: hrtf_dir/Subject_002/HRIR_44100/azi_*.wav
    sample_rate : int
        Audio pipeline sample rate. Must match SADIE II IR sample rate (44100).

    Example
    -------
    >>> renderer = HRTFRenderer("/backend/hrtf/sadie2", sample_rate=44100)
    >>> stereo_chunk = renderer.render(mono_chunk, speaker_id="SPK_0")
    >>> # stereo_chunk.shape == (len(mono_chunk) + ir_len - 1, 2)
    """

    def __init__(self, hrtf_dir: str | Path, sample_rate: int = 44100) -> None:
        self.hrtf_dir   = Path(hrtf_dir)
        self.sample_rate = sample_rate

        # Dict[azimuth_degrees -> (left_IR_array, right_IR_array)]
        # Populated in TODO block below.
        self._ir_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

        # Sorted list of available azimuths — used by _nearest_azimuth()
        self._available_azimuths: list[float] = []

        self._load_impulse_responses()
        logger.info(
            "HRTFRenderer ready — %d azimuths loaded from %s",
            len(self._ir_cache),
            self.hrtf_dir,
        )

    # ------------------------------------------------------------------
    # Initialisation  (TEAMMATE: implement this)
    # ------------------------------------------------------------------

    def _load_impulse_responses(self) -> None:
        """Walk hrtf_dir and load all azimuth IRs into self._ir_cache.

        TODO:
          1. Glob for files matching Subject_002/HRIR_44100/azi_*_ele_0.wav
          2. For each file, parse azimuth from filename (e.g. azi_330_ele_0 → 330.0)
          3. Load with scipy.io.wavfile.read() or soundfile.read()
          4. Split into left_ir (ch 0) and right_ir (ch 1) — both shape (N,)
          5. Normalise to float32 in range [-1, 1]
          6. Store: self._ir_cache[azimuth] = (left_ir, right_ir)
          7. Set self._available_azimuths = sorted(self._ir_cache.keys())
          8. Assert len(self._ir_cache) > 0 or raise FileNotFoundError
        """
        raise NotImplementedError(
            "Teammate: implement _load_impulse_responses() — "
            "see module docstring for SADIE II file layout."
        )

    # ------------------------------------------------------------------
    # Azimuth lookup
    # ------------------------------------------------------------------

    def _nearest_azimuth(self, target: float) -> float:
        """Return the measured azimuth in self._ir_cache closest to *target*.

        Uses circular distance so 359° is correctly close to 0°.

        TODO:
          Compute circular distance for each available azimuth:
            dist = min(abs(a - target), 360 - abs(a - target))
          Return the azimuth with minimum distance.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public render method
    # ------------------------------------------------------------------

    def render(self, chunk: np.ndarray, speaker_id: str) -> np.ndarray:
        """Spatialise a mono audio chunk for the given speaker.

        Parameters
        ----------
        chunk : np.ndarray
            Mono float32 audio samples, shape (N,).
            N is typically CHUNK_SIZE from config.py (e.g. 960 samples).
        speaker_id : str
            "SPK_0" or "SPK_1". Azimuth is looked up from SPEAKER_AZIMUTHS.

        Returns
        -------
        np.ndarray
            Stereo float32 array, shape (N + ir_len - 1, 2).
            Column 0 = left ear, column 1 = right ear.

        Raises
        ------
        ValueError
            If speaker_id is not in SPEAKER_AZIMUTHS.

        TODO:
          1. Validate speaker_id is in SPEAKER_AZIMUTHS
          2. Get target azimuth from SPEAKER_AZIMUTHS[speaker_id]
          3. nearest = self._nearest_azimuth(target)
          4. left_ir, right_ir = self._ir_cache[nearest]
          5. from scipy.signal import fftconvolve
             left_out  = fftconvolve(chunk, left_ir,  mode='full').astype(np.float32)
             right_out = fftconvolve(chunk, right_ir, mode='full').astype(np.float32)
          6. stereo = np.stack([left_out, right_out], axis=1)
          7. return stereo
        """
        raise NotImplementedError
