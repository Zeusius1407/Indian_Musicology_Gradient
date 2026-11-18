"""Neural-vocoder resynthesis with a pretrained BigVGAN.

BigVGAN is HiFi-GAN's generator with anti-aliased periodic (Snake)
activations, trained on speech, music and environmental sound rather than
speech alone. That last point is why it is used here instead of a stock
HiFi-GAN: every public HiFi-GAN checkpoint is a 22 kHz speech model and turns a
mridangam into a buzzing voice. ``bigvgan_v2_44khz_128band_512x`` also runs
at 44.1 kHz with a 2048-point window and a 512-sample hop -- the pipeline's
own STFT settings -- so masks need no resampling in frequency, only a
half-frame shift in time (see ``_masks_on_vocoder_grid``).

No fine-tuning: the vocoder never saw Carnatic audio, and it is fed masked
*estimates*, not clean mels. Both cost SDR; the comparison in
``cli/compare_resynthesis.py`` measures how much.

The model code ships inside the Hugging Face repo rather than on PyPI, so
``_import_bigvgan`` downloads it and imports it from the snapshot. Only load
repos you trust: this executes their Python.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import torch

from ..dsp.stft import STFTConfig

__all__ = ["Vocoder"]

# Top-level module names inside the BigVGAN snapshot. They are generic enough
# to collide with other packages, so they are imported in isolation.
_SNAPSHOT_MODULES = ("bigvgan", "activations", "utils", "env", "meldataset", "alias_free_activation")


def _import_bigvgan(repo: str):
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo, allow_patterns=["*.py", "config.json", "bigvgan_generator.pt"])

    def owned(name: str) -> bool:
        return name.split(".", 1)[0] in _SNAPSHOT_MODULES

    stashed = {k: sys.modules.pop(k) for k in list(sys.modules) if owned(k)}
    sys.path.insert(0, path)
    try:
        module = importlib.import_module("bigvgan")
    finally:
        sys.path.remove(path)
        # Drop the snapshot's modules (``module`` keeps what it needs) and put
        # back whatever held those names before.
        for k in [k for k in sys.modules if owned(k)]:
            del sys.modules[k]
        sys.modules.update(stashed)
    return module, path


class Vocoder:
    """Masks + mixture in, one waveform per source out.

    Parameters
    ----------
    repo:
        Hugging Face repo of a BigVGAN v2 checkpoint.
    device:
        Torch device; defaults to CUDA when available.
    chunk_frames:
        Mel frames generated per forward pass (512 = ~6 s at 44.1 kHz), which
        bounds GPU memory on concert-length input.
    context_frames:
        Extra frames on each side of a chunk, generated and then discarded so
        chunk seams fall outside the generator's receptive field.
    """

    def __init__(
        self,
        repo: str = "nvidia/bigvgan_v2_44khz_128band_512x",
        device: str | torch.device | None = None,
        chunk_frames: int = 512,
        context_frames: int = 32,
    ):
        module, path = _import_bigvgan(repo)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = module.BigVGAN.from_pretrained(path, use_cuda_kernel=False)
        model.remove_weight_norm()
        self.model = model.eval().to(self.device)
        self.h = model.h
        self.chunk_frames = chunk_frames
        self.context_frames = context_frames

        import librosa

        mel = librosa.filters.mel(
            sr=self.h.sampling_rate, n_fft=self.h.n_fft, n_mels=self.h.num_mels, fmin=self.h.fmin, fmax=self.h.fmax
        )
        self.mel_basis = torch.from_numpy(mel).float().to(self.device)
        self.window = torch.hann_window(self.h.win_size).to(self.device)

    def _check(self, stft_cfg: STFTConfig) -> None:
        if stft_cfg.sample_rate != self.h.sampling_rate or stft_cfg.n_fft != self.h.n_fft:
            raise ValueError(
                f"vocoder runs at {self.h.sampling_rate} Hz with n_fft={self.h.n_fft}; the masks were made at "
                f"{stft_cfg.sample_rate} Hz with n_fft={stft_cfg.n_fft}"
            )

    def _magnitude(self, audio: torch.Tensor) -> torch.Tensor:
        """``(F, N)`` magnitude exactly as BigVGAN's training mels were computed."""
        pad = (self.h.n_fft - self.h.hop_size) // 2
        y = torch.nn.functional.pad(audio[None, None], (pad, pad), mode="reflect")[0]
        spec = torch.stft(
            y,
            self.h.n_fft,
            hop_length=self.h.hop_size,
            win_length=self.h.win_size,
            window=self.window,
            center=False,
            return_complex=True,
        )[0]
        return torch.sqrt(spec.real**2 + spec.imag**2 + 1e-9)

    def _masks_on_vocoder_grid(self, masks: np.ndarray, stft_cfg: STFTConfig, n_frames: int) -> torch.Tensor:
        """Linearly interpolate ``(S, F, frames)`` masks onto the vocoder's frame times.

        Pipeline frame ``k`` is centred on sample ``k * hop`` (``dsp.stft``
        zero-pads half a window); vocoder frame ``k`` is centred on
        ``k * hop + hop / 2``. With equal hops that is the mean of two
        neighbouring pipeline frames.
        """
        centres = np.arange(n_frames) * self.h.hop_size + (self.h.n_fft // 2 - (self.h.n_fft - self.h.hop_size) // 2)
        pos = np.clip(centres / stft_cfg.hop_length, 0, masks.shape[-1] - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, masks.shape[-1] - 1)
        frac = (pos - lo).astype(np.float32)
        grid = masks[..., lo] * (1.0 - frac) + masks[..., hi] * frac
        return torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float32)).to(self.device)

    @torch.no_grad()
    def _generate(self, log_mel: torch.Tensor) -> torch.Tensor:
        """``(M, N)`` log-mel -> ``N * hop`` samples, chunk by chunk."""
        n, c, hop = log_mel.shape[-1], self.context_frames, self.h.hop_size
        out = []
        for a in range(0, n, self.chunk_frames):
            b = min(n, a + self.chunk_frames)
            lo, hi = max(0, a - c), min(n, b + c)
            wav = self.model(log_mel[None, :, lo:hi])[0, 0]
            out.append(wav[(a - lo) * hop : (b - lo) * hop])
        return torch.cat(out)

    def from_masks(self, masks: np.ndarray, mixture: np.ndarray, stft_cfg: STFTConfig) -> np.ndarray:
        """Vocode ``mask * |mixture|`` for every source. Returns ``(S, T)``."""
        self._check(stft_cfg)
        n = len(mixture)
        # BigVGAN was trained on peak-normalized audio and the log-mel is not
        # scale-invariant, so vocode at that level and undo the gain after.
        gain = 0.95 / max(float(np.max(np.abs(mixture))), 1e-8)
        audio = torch.from_numpy(np.asarray(mixture, dtype=np.float32) * gain).to(self.device)
        mag = self._magnitude(audio)
        grid = self._masks_on_vocoder_grid(masks, stft_cfg, mag.shape[-1])

        stems = []
        for m in grid:
            log_mel = torch.log(torch.clamp(self.mel_basis @ (m * mag), min=1e-5))
            wav = self._generate(log_mel).cpu().numpy() / gain
            stems.append(np.pad(wav, (0, max(0, n - len(wav))))[:n])
        return np.stack(stems).astype(np.float32)
