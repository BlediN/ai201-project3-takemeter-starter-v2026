#!/usr/bin/env python3
"""
Convert the saved PyTorch model (takemeter_model/) to ONNX for hosting.

    python export_onnx.py

Why: Render's free tier has 512 MB of memory. Serving the PyTorch model peaked
at 613 MB (measured), so the hosted version runs the same weights through
onnxruntime instead of torch. This script writes takemeter_onnx/ and then
checks the two models agree on every post in test_split.csv. If they don't
agree, it exits 1 and the ONNX model shouldn't be deployed.

Run it locally, once, after the notebook's save cell. The server never needs
torch.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SRC = Path("takemeter_model")
DST = Path("takemeter_onnx")
CHECK_CSV = "test_split.csv"


def export():
    if not (SRC / "model.safetensors").exists():
        sys.exit(f"error: {SRC}/ not found. Run the notebook's save cell first.")

    tokenizer = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC).eval()
    meta = json.loads((SRC / "takemeter_meta.json").read_text(encoding="utf-8"))

    DST.mkdir(exist_ok=True)
    dummy = tokenizer("a sample post", return_tensors="pt")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(DST / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    tokenizer.save_pretrained(DST)
    # The label order lives in the PyTorch config; the server reads it from here.
    meta["labels"] = [model.config.id2label[i] for i in range(model.config.num_labels)]
    (DST / "takemeter_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for extra in ("config.json",):
        shutil.copy(SRC / extra, DST / extra)
    return tokenizer, model, meta


def check(tokenizer, model, meta):
    """Same posts through both models. Labels must match; report the max prob gap."""
    import onnxruntime as ort
    import pandas as pd

    session = ort.InferenceSession(str(DST / "model.onnx"),
                                   providers=["CPUExecutionProvider"])
    posts = pd.read_csv(CHECK_CSV)["text"].tolist()
    same, max_gap = 0, 0.0
    for text in posts:
        enc = tokenizer(text, truncation=True, max_length=meta["max_length"],
                        return_tensors="np")
        onnx_logits = session.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                               "attention_mask": enc["attention_mask"].astype(np.int64)})[0][0]
        with torch.no_grad():
            pt_logits = model(input_ids=torch.tensor(enc["input_ids"]),
                              attention_mask=torch.tensor(enc["attention_mask"])).logits[0].numpy()
        softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        same += int(onnx_logits.argmax() == pt_logits.argmax())
        max_gap = max(max_gap, float(np.abs(softmax(onnx_logits) - softmax(pt_logits)).max()))

    size_mb = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {DST}/ ({size_mb:.0f} MB)")
    print(f"labels match on {same}/{len(posts)} posts in {CHECK_CSV}")
    print(f"largest probability difference: {max_gap:.2e}")
    return same == len(posts)


if __name__ == "__main__":
    ok = check(*export())
    sys.exit(0 if ok else 1)
