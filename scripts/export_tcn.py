#!/usr/bin/env python
"""Export a full-sequence-trained TCN checkpoint as a windowed ONNX graph.

Training uses the full-sequence form; the submission uses the windowed form. The
weights are identical and the network is causal, so the two agree -- but "agree
in principle" is not evidence, so this verifies three things before writing
anything:

  1. the receptive field still fits inside the context window;
  2. full-sequence and windowed torch outputs match on real data;
  3. the exported ONNX graph matches the torch windowed model.

    python scripts/export_tcn.py --checkpoint artifacts/tcn_long.pt
    python scripts/export_tcn.py --checkpoint artifacts/tcn_long.pt --install
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lobkit.data import LobCorpus  # noqa: E402
from lobkit.fullseq import assert_equivalence_safe, check_numerical_equivalence  # noqa: E402
from lobkit.tcn import TCN, build_from_checkpoint  # noqa: E402

DEFAULT_CORPUS = Path.home() / ".cache" / "lob-prediction" / "corpus"


class _DualMode(torch.nn.Module):
    """Routes by sequence length so one module can be probed both ways."""

    def __init__(self, model: TCN, context_len: int):
        super().__init__()
        self.model = model
        self.context_len = context_len

    def forward(self, x):
        self.model.full_sequence = x.shape[1] != self.context_len
        return self.model(x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--checkpoint", default=str(root / "artifacts" / "tcn_long.pt"))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--out", default=None, help="Defaults to the checkpoint name with .onnx")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Also copy into submission_safe_3model/ as improved_tcn.onnx, replacing "
        "the shipped model. The previous file is backed up alongside.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"missing {ckpt_path}", file=sys.stderr)
        return 1

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    context_len = int(checkpoint["context_len"])
    dilations = list(checkpoint["dilations"])
    print(
        f"checkpoint: epoch {checkpoint['epoch']}, inner-val weighted Pearson "
        f"{checkpoint['best_score']:+.5f}"
    )

    lookback = assert_equivalence_safe(dilations, context_len)
    print(f"gate 1/3  receptive field {lookback} <= context {context_len}  OK")

    model = build_from_checkpoint(checkpoint, full_sequence=True).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    corpus = LobCorpus(args.corpus)
    probe = np.ascontiguousarray(corpus.raw[0], dtype=np.float32)
    steps = np.array([99, 100, 126, 127, 128, 300, 700, 999])
    diff = check_numerical_equivalence(
        _DualMode(model, context_len), probe, steps, context_len, tolerance=args.tolerance
    )
    print(f"gate 2/3  max|full-sequence - windowed| = {diff:.3e}  OK")

    onnx_path = Path(args.out) if args.out else ckpt_path.with_suffix(".onnx")
    export_model = build_from_checkpoint(checkpoint, full_sequence=False).eval()
    for module in export_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    dummy = torch.zeros(1, context_len, 32, dtype=torch.float32)
    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnxruntime as ort

    sample = torch.randn(8, context_len, 32, dtype=torch.float32)
    with torch.no_grad():
        torch_out = export_model(sample).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["output"], {"input": sample.numpy()})[0]
    parity = float(np.max(np.abs(torch_out - onnx_out)))
    if parity > args.tolerance:
        print(f"gate 3/3  ONNX parity {parity:.3e} exceeds {args.tolerance:.1e}  FAIL", file=sys.stderr)
        return 1
    print(f"gate 3/3  max|torch - onnx| = {parity:.3e}  OK")

    config_path = onnx_path.with_suffix(".json")
    config_path.write_text(
        json.dumps(
            {"onnx_path": onnx_path.name, "context_len": context_len, "raw_dim": 32}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {onnx_path} ({onnx_path.stat().st_size / 1e3:.0f} kB)")
    print(f"wrote {config_path}")

    if args.install:
        sub = root / "submission_safe_3model"
        target = sub / "improved_tcn.onnx"
        if target.exists():
            backup = sub / "improved_tcn.onnx.orig"
            if not backup.exists():
                shutil.copy2(target, backup)
                print(f"backed up original to {backup}")
        shutil.copy2(onnx_path, target)
        shutil.copy2(config_path, sub / "improved_tcn.json")
        (sub / "improved_tcn.json").write_text(
            json.dumps(
                {"onnx_path": "improved_tcn.onnx", "context_len": context_len, "raw_dim": 32},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"installed into {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
