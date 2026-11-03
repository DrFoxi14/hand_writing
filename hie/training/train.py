from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from hie.device import detect_device
from hie.training.data import LineDataset
from hie.training.model import WriterConditionedGenerator
from hie.config import ANNOTATIONS, CHECKPOINTS


def train(manifest: Path, output: Path, epochs: int = 30, lr: float = 2e-4,
          min_lines: int = 40, device_name: str | None = None) -> dict:
    rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    page_splits = {}
    for row in rows:
        page_id=row.get("page_id")
        split=row.get("split", "train")
        if page_id and page_id in page_splits and page_splits[page_id] != split:
            raise ValueError(f"Page {page_id} appears in more than one data split; split by page/session")
        if page_id: page_splits[page_id]=split
    training_rows = [r for r in rows if r.get("split", "train") == "train"]
    if len(training_rows) < min_lines:
        raise ValueError(f"Training requires at least {min_lines} annotated line crops; found {len(training_rows)}")
    validation_rows = [r for r in rows if r.get("split") == "validation"]
    if not validation_rows:
        raise ValueError("A held-out validation split is required; reserve at least one complete page/session before training")
    chars = sorted({c for r in training_rows for c in r["text"]})
    vocab = {c:i+1 for i,c in enumerate(chars)}
    ds = LineDataset(manifest, "train", vocab=vocab)
    val_ds = LineDataset(manifest, "validation", vocab=vocab)
    device = device_name or detect_device()["device"]
    model = WriterConditionedGenerator(vocab_size=len(vocab)+1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    model.train()
    for epoch in range(epochs):
        losses = []
        for sample in ds:
            chars_tensor = sample["chars"].unsqueeze(0).to(device)
            style = sample["style"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            prediction = model(chars_tensor, style, output_height=target.shape[2], pixels_per_char=20)
            if prediction.shape[-1] != target.shape[-1]:
                prediction = F.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses=[]
        with torch.inference_mode():
            for sample in val_ds:
                chars_tensor=sample["chars"].unsqueeze(0).to(device)
                style=sample["style"].unsqueeze(0).to(device)
                target=sample["target"].unsqueeze(0).to(device)
                prediction=model(chars_tensor,style,output_height=target.shape[2],pixels_per_char=20)
                if prediction.shape[-1] != target.shape[-1]:
                    prediction=F.interpolate(prediction,size=target.shape[-2:],mode="bilinear",align_corners=False)
                val_losses.append(float(F.binary_cross_entropy(prediction,target).cpu()))
        history.append({"epoch":epoch+1,"loss":sum(losses)/max(1,len(losses)),
                        "validation_loss":sum(val_losses)/max(1,len(val_losses))})
        model.train()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"vocab":vocab,"epochs":epochs,
                "architecture":"WriterConditionedGenerator-v1","training_samples":len(ds)}, output)
    metrics = {"checkpoint":str(output),"device":device,"samples":len(ds),"validation_samples":len(val_ds),"vocab_size":len(vocab),"history":history}
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train HIE's writer-conditioned line generator")
    parser.add_argument("--manifest", type=Path, default=ANNOTATIONS / "lines.jsonl")
    parser.add_argument("--output", type=Path, default=CHECKPOINTS / "writer_v1.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-lines", type=int, default=40)
    args = parser.parse_args()
    result = train(args.manifest,args.output,args.epochs,min_lines=args.min_lines)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
