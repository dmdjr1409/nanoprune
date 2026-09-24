import inspect
import warnings
from pathlib import Path
from typing import Union


def _all_heads_module(model):
    import torch.nn as nn

    class _AllHeads(nn.Module):
        """Exports relevance, choice and score outputs in a single graph."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            out = self.inner.forward_all(input_ids, attention_mask)
            return out["relevance_logits"], out["relevance_probs"], out["choice_probs"], out["score"]

    return _AllHeads(model)


def inline_external_data(path: Union[str, Path]) -> bool:
    """Re-save an ONNX file with its weights embedded (needs the ``onnx`` package).

    Recent PyTorch versions write weights to a separate ``.onnx.data`` file by
    default; a single self-contained file is easier to ship. Returns True when
    the file was rewritten.
    """
    try:
        import onnx
    except ImportError:
        return False
    path = Path(path)
    shallow = onnx.load(str(path), load_external_data=False)
    external = {
        entry.value
        for tensor in shallow.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        for entry in tensor.external_data
        if entry.key == "location"
    }
    if not external:
        return False
    full = onnx.load(str(path))
    onnx.save_model(full, str(path), save_as_external_data=False)
    for name in external:
        data_file = path.parent / name
        if data_file.exists():
            data_file.unlink()
    return True


def export_to_onnx(
    model,
    output_path: Union[str, Path],
    seq_len: int = 256,
    quantize: bool = False,
    verbose: bool = False,
    heads: str = "relevance",
    opset_version: int = 17,
) -> Path:
    """
    Exports a PyTorch NanoPruneModel to a self-contained ONNX graph.

    Args:
        heads: ``"relevance"`` exports (logits, probabilities); ``"all"`` also
            exports ``choice_probs`` and ``score`` so the ONNX backend can serve
            every primitive.
    """
    import torch

    if heads not in ("relevance", "all"):
        raise ValueError("heads must be 'relevance' or 'all'")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab = model.token_embeddings.num_embeddings
    dummy_input_ids = torch.randint(5, max(6, min(vocab, 100)), (2, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((2, seq_len), dtype=torch.long)

    was_training = model.training
    model.eval()
    module = model if heads == "relevance" else _all_heads_module(model)
    # torch.onnx.export restores the wrapper's own mode afterwards, which would
    # otherwise switch the wrapped model back to training mode (dropout on).
    module.eval()
    output_names = ["logits", "probabilities"] + (["choice_probs", "score"] if heads == "all" else [])
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
    }
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=verbose,
    )
    supported = inspect.signature(torch.onnx.export).parameters
    # The TorchScript exporter handles dynamic_axes and needs no extra package;
    # newer PyTorch versions default to the dynamo exporter and external data.
    if "dynamo" in supported:
        kwargs["dynamo"] = False
    if "external_data" in supported:
        kwargs["external_data"] = False

    # The fused inference kernel of nn.TransformerEncoderLayer has no ONNX
    # equivalent: make sure the exporter traces the regular layers.
    mha = getattr(torch.backends, "mha", None)
    fastpath = mha.get_fastpath_enabled() if mha is not None else None
    if mha is not None:
        mha.set_fastpath_enabled(False)
    try:
        with warnings.catch_warnings():
            # Tracing warnings about shape checks inside nn.MultiheadAttention are expected.
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            torch.onnx.export(module, (dummy_input_ids, dummy_mask), str(output_path), **kwargs)
    finally:
        if mha is not None:
            mha.set_fastpath_enabled(fastpath)
        model.train(was_training)
    inline_external_data(output_path)

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quant_path = output_path.with_suffix(".quant.onnx")
            quantize_dynamic(
                model_input=str(output_path),
                model_output=str(quant_path),
                weight_type=QuantType.QUInt8,
            )
            return quant_path
        except ImportError:
            pass

    return output_path
