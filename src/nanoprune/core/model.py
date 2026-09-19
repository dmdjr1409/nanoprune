import math
from typing import Optional, Tuple

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

if HAS_TORCH:
    class NanoPruneModel(nn.Module):
        """
        NanoPrune: 2-layer Transformer Encoder (~960k parameters)
        Single-pass calibrated relevance scoring for RAG context pruning.
        """
        def __init__(
            self,
            vocab_size: int = 4096,
            d_model: int = 128,
            n_heads: int = 4,
            d_ff: int = 512,
            n_layers: int = 2,
            max_seq_len: int = 256,
            dropout: float = 0.1,
            temperature: float = 1.0,
            num_choices: int = 4,
        ):
            super().__init__()
            self.d_model = d_model
            self.max_seq_len = max_seq_len
            self.num_choices = num_choices
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
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            # Primitive 1: Calibrated relevance / prune head
            self.relevance_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            self.head = self.relevance_head

            # Primitive 2: Multi-class choice head
            self.choice_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_choices),
            )

            # Primitive 3: Continuous rubric score head
            self.score_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

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
                probabilities: Calibrated relevance probabilities (batch_size, 1)
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
