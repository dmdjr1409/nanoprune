import math
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

# Heads that older checkpoints may not contain (v0.1/v0.2 only had the relevance head).
OPTIONAL_HEAD_PREFIXES = ("choice_head.", "score_head.")


def infer_architecture(state_dict: Mapping[str, Any]) -> Dict[str, int]:
    """Read the architecture hyper-parameters stored implicitly in a state dict.

    The number of attention heads cannot be recovered from the weights; it is
    returned only through the ``n_heads`` default rule used by every released
    checkpoint (8 heads for d_model >= 256, 4 otherwise).
    """
    emb = state_dict["token_embeddings.weight"]
    vocab_size, d_model = int(emb.shape[0]), int(emb.shape[1])
    layer_ids = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("encoder.layers.") and key.split(".")[2].isdigit()
    }
    head_key = "relevance_head.0.weight" if "relevance_head.0.weight" in state_dict else "head.0.weight"
    arch = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": len(layer_ids),
        "d_ff": int(state_dict["encoder.layers.0.linear1.weight"].shape[0]),
        "max_seq_len": int(state_dict["position_embeddings.weight"].shape[0]),
        "head_hidden": int(state_dict[head_key].shape[0]),
        "n_heads": 8 if d_model >= 256 else 4,
        "num_choices": int(state_dict["choice_head.3.weight"].shape[0]) if "choice_head.3.weight" in state_dict else 4,
    }
    return arch


if HAS_TORCH:
    class NanoPruneModel(nn.Module):
        """
        NanoPrune v0.4: 4-layer Transformer encoder (~5.45M parameters
        with the 8192-token WordPiece vocabulary and multi-head outputs).
        Single-pass relevance scoring for RAG context pruning.
        """
        def __init__(
            self,
            vocab_size: int = 8192,
            d_model: int = 256,
            n_heads: int = 8,
            d_ff: int = 1024,
            n_layers: int = 4,
            max_seq_len: int = 256,
            dropout: float = 0.1,
            temperature: float = 1.0,
            num_choices: int = 4,
            head_hidden: int = 128,
        ):
            super().__init__()
            self.d_model = d_model
            self.max_seq_len = max_seq_len
            self.num_choices = num_choices
            self._config = {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "d_ff": d_ff,
                "n_layers": n_layers,
                "max_seq_len": max_seq_len,
                "num_choices": num_choices,
                "head_hidden": head_hidden,
            }
            self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=False)

            self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.position_embeddings = nn.Embedding(max_seq_len, d_model)
            self.layer_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

            # Primitive 1: relevance / prune head
            self.relevance_head = nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            # Legacy alias kept so that "head.*" keys stay in saved state dicts.
            self.head = self.relevance_head

            # Primitive 2: multi-class choice head
            self.choice_head = nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_choices),
            )

            # Primitive 3: continuous rubric score head
            self.score_head = nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

        def config(self) -> Dict[str, int]:
            """Architecture hyper-parameters, as written to model manifests."""
            return dict(self._config)

        @classmethod
        def from_state_dict(
            cls,
            state_dict: Mapping[str, Any],
            n_heads: Optional[int] = None,
        ) -> "NanoPruneModel":
            """Build a model matching ``state_dict`` and load it.

            Loading is strict except for the choice/score heads, which legacy
            checkpoints do not contain, and the temperature buffer.
            """
            state = dict(state_dict)
            # Legacy checkpoints named the relevance head "head".
            for key in list(state):
                if key.startswith("head."):
                    state.setdefault("relevance_" + key, state[key])
            for key in list(state):
                if key.startswith("relevance_head."):
                    state.setdefault(key[len("relevance_"):], state[key])

            arch = infer_architecture(state)
            if n_heads is not None:
                arch["n_heads"] = n_heads
            model = cls(**arch)
            result = model.load_state_dict(state, strict=False)
            missing = [
                key for key in result.missing_keys
                if not key.startswith(OPTIONAL_HEAD_PREFIXES) and key != "temperature"
            ]
            if missing or result.unexpected_keys:
                raise RuntimeError(
                    "Checkpoint does not match the NanoPrune architecture "
                    f"(missing: {missing[:5]}, unexpected: {list(result.unexpected_keys)[:5]})"
                )
            model.loaded_heads = {
                "relevance": True,
                "choice": not any(k.startswith("choice_head.") for k in result.missing_keys),
                "score": not any(k.startswith("score_head.") for k in result.missing_keys),
            }
            model.eval()
            return model

        def _encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            batch_size, seq_len = input_ids.size()
            positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

            x = self.token_embeddings(input_ids) * math.sqrt(self.d_model)
            x = x + self.position_embeddings(positions)
            x = self.layer_norm(x)
            x = self.dropout(x)

            padding_mask = None
            if attention_mask is not None:
                padding_mask = attention_mask == 0

            return self.encoder(x, src_key_padding_mask=padding_mask)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                input_ids: Tensor of shape (batch_size, seq_len)
                attention_mask: Tensor of shape (batch_size, seq_len), 1 for valid, 0 for pad
            Returns:
                logits: Raw relevance logits (batch_size, 1)
                probabilities: Temperature-scaled relevance probabilities (batch_size, 1)
            """
            encoded = self._encode(input_ids, attention_mask)
            cls_rep = encoded[:, 0, :]
            logits = self.relevance_head(cls_rep)

            scaled_logits = logits / torch.clamp(self.temperature, min=1e-3)
            probabilities = torch.sigmoid(scaled_logits)

            return logits, probabilities

        def forward_all(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> dict:
            """
            Multi-primitive forward pass returning:
            - relevance_logits: (batch_size, 1)
            - relevance_probs: (batch_size, 1)
            - choice_logits: (batch_size, num_choices)
            - choice_probs: (batch_size, num_choices)
            - score: (batch_size, 1)
            """
            encoded = self._encode(input_ids, attention_mask)
            cls_rep = encoded[:, 0, :]

            # Relevance
            rel_logits = self.relevance_head(cls_rep)
            rel_probs = torch.sigmoid(rel_logits / torch.clamp(self.temperature, min=1e-3))

            # Choice
            choice_logits = self.choice_head(cls_rep)
            choice_probs = F.softmax(choice_logits, dim=-1)

            # Score (scaled to [0, 4])
            score = torch.sigmoid(self.score_head(cls_rep)) * 4.0

            return {
                "relevance_logits": rel_logits,
                "relevance_probs": rel_probs,
                "choice_logits": choice_logits,
                "choice_probs": choice_probs,
                "score": score,
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:
    class NanoPruneModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required to initialize NanoPruneModel. "
                "Install it with: pip install torch"
            )

        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            raise ImportError(
                "PyTorch is required to load .pt checkpoints. "
                "Install it with: pip install torch"
            )
