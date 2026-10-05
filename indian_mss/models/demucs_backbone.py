"""HTDemucs, re-targeted from Western stems to the Sanidha instruments.

Transfer strategy
-----------------
The pretrained ``htdemucs`` predicts ``drums, bass, other, vocals``. Rather
than bolting a randomly initialised head onto it, the outermost decoder layers
(the only layers whose output channels are per-source) are rebuilt with one
block of channels per *new* source, each copied from the closest Western
source in ``WESTERN_INIT``:

    vocal     <- vocals   (a voice is a voice)
    mridangam <- drums    (pitched, but still a membrane transient)
    ghatam    <- drums    (clay pot: transient, broadband)
    violin    <- other    (sustained melodic residue)

So the model starts training already roughly separating voice from percussion
from melody, and fine-tuning only has to teach it what is *Indian* about each
timbre. Everything else -- both encoders, the cross-domain transformer -- is
pretrained and starts frozen (parameter-efficient transfer, see
``trainable_groups``).

Both branches of HTDemucs lay out output channels source-major
(``view(B, S, -1, ...)`` in ``HTDemucs.forward``), which is what makes this
block copy correct; ``tests/test_models.py`` checks the re-targeted model's
outputs equal the original's, source for source.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["DemucsSeparator", "WESTERN_INIT", "load_htdemucs", "retarget_sources"]

#: Which pretrained Western source initialises each Indian instrument.
WESTERN_INIT = {
    "vocal": "vocals",
    "violin": "other",
    "mridangam": "drums",
    "ghatam": "drums",
    "tabla": "drums",
    "veena": "other",
    "sitar": "other",
    "sarod": "other",
    "flute": "other",
}


def load_htdemucs(name: str = "htdemucs") -> nn.Module:
    """Download (first use only) and return a single pretrained HTDemucs.

    Bags of several models (``htdemucs_ft``) are refused: each member is a
    specialist for one Western stem, which is exactly the prior we are trying
    to replace, and fine-tuning four models costs four times the memory.
    """
    from demucs.pretrained import get_model

    bag = get_model(name)
    models = getattr(bag, "models", [bag])
    if len(models) != 1:
        raise ValueError(f"{name!r} is a bag of {len(models)} models; use a single model such as 'htdemucs'")
    return models[0]


def _copy_source_blocks(conv: nn.Module, old_sources: list[str], picks: list[int]) -> None:
    """Rebuild a transposed conv's output channels, one block per new source."""
    weight = conv.weight.data  # (chin, chout, *kernel) for ConvTranspose
    per = weight.shape[1] // len(old_sources)
    new_w = torch.cat([weight[:, i * per : (i + 1) * per] for i in picks], dim=1)
    conv.weight = nn.Parameter(new_w.clone())
    if conv.bias is not None:
        b = conv.bias.data
        conv.bias = nn.Parameter(torch.cat([b[i * per : (i + 1) * per] for i in picks]).clone())
    conv.out_channels = new_w.shape[1]


def retarget_sources(model: nn.Module, sources: list[str], init: dict[str, str] | None = None) -> nn.Module:
    """Re-target an HTDemucs in place to predict ``sources``.

    ``init`` maps each new source to the existing source whose output weights
    it starts from; defaults to ``WESTERN_INIT``, and a source already in the
    model maps to itself.
    """
    init = {**WESTERN_INIT, **(init or {})}
    old = list(model.sources)
    picks = []
    for s in sources:
        donor = s if s in old else init.get(s)
        if donor not in old:
            raise ValueError(f"no initialisation for source {s!r}: add it to WESTERN_INIT (model has {old})")
        picks.append(old.index(donor))

    for branch in (model.decoder, model.tdecoder):
        _copy_source_blocks(branch[-1].conv_tr, old, picks)
    model.sources = list(sources)

    # Keep the captured constructor arguments in sync, so the model can be
    # rebuilt from a checkpoint without downloading the pretrained weights.
    args, kwargs = model._init_args_kwargs
    if "sources" in kwargs:
        kwargs = {**kwargs, "sources": list(sources)}
    else:
        args = (list(sources), *args[1:])
    model._init_args_kwargs = (args, kwargs)
    return model


class DemucsSeparator(nn.Module):
    """Mono-in, mono-out wrapper around a (stereo) HTDemucs.

    The whole pipeline is mono (see ``dsp.stft.stft``); HTDemucs is stereo.
    The mono mix is duplicated to both channels and the two output channels
    are averaged.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def sources(self) -> list[str]:
        return list(self.model.sources)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` -> ``(B, S, T)``."""
        x = mix.unsqueeze(1).expand(-1, self.model.audio_channels, -1)
        return self.model(x).mean(dim=2)

    def head_parameters(self) -> list[nn.Parameter]:
        """The re-targeted output layers -- the only weights trained in phase 1."""
        return [
            *self.model.decoder[-1].parameters(),
            *self.model.tdecoder[-1].parameters(),
        ]

    def trainable_groups(self, phase: int, lr: float, encoder_lr: float, unfreeze: list[str]) -> list[dict]:
        """Freeze everything, then open up what this training phase may touch.

        Phase 1: only the head, at ``lr``. Phase 2: additionally the top-level
        submodules named in ``unfreeze`` (e.g. ``decoder``, ``tdecoder``,
        ``crosstransformer``), at the 10x lower ``encoder_lr`` so the
        pretrained features are adjusted rather than overwritten. Modules not
        listed -- by default both encoders -- stay frozen throughout.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        head = self.head_parameters()
        head_ids = {id(p) for p in head}
        for p in head:
            p.requires_grad_(True)
        groups = [{"params": head, "lr": lr, "name": "head"}]
        if phase >= 2:
            known = dict(self.model.named_children())
            missing = [n for n in unfreeze if n not in known]
            if missing:
                raise ValueError(f"cannot unfreeze {missing}: HTDemucs has {sorted(known)}")
            body = [p for n in unfreeze for p in known[n].parameters() if id(p) not in head_ids]
            for p in body:
                p.requires_grad_(True)
            if body:
                groups.append({"params": body, "lr": encoder_lr, "name": "body"})
        return groups
