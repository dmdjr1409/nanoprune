import os
from pathlib import Path
from typing import Union, Optional

def export_to_onnx(
    model,
    output_path: Union[str, Path],
    seq_len: int = 256,
    quantize: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Exports a PyTorch NanoPruneModel to a compact ONNX graph.
    """
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input_ids = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((1, seq_len), dtype=torch.long)

    model.eval()

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "probabilities"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

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
