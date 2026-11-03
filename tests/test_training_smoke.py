import json
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

torch=pytest.importorskip("torch")
from hie.training.train import train
from hie.inference.neural import generate_from_checkpoint


def make_crop(path, char):
    im=Image.new("L",(100,40),255);d=ImageDraw.Draw(im)
    d.line((15,12,30,30),fill=0,width=3)
    if char=="b":d.line((30,12,30,30),fill=0,width=3)
    im.save(path)


def test_tiny_training_run_and_checkpoint_inference(tmp_path):
    rows=[]
    for i,(char,split) in enumerate([("a","train"),("b","train"),("a","validation")]):
        path=tmp_path/f"line_{i}.png";make_crop(path,char)
        rows.append({"id":str(i),"page_id":f"page_{i}","image":str(path),"text":char,"split":split})
    manifest=tmp_path/"lines.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    checkpoint=tmp_path/"model.pt"
    result=train(manifest,checkpoint,epochs=1,min_lines=2,device_name="cpu")
    assert checkpoint.exists()
    assert result["validation_samples"]==1
    image=generate_from_checkpoint("ab",tmp_path/"line_0.png",checkpoint,device="cpu")
    assert image.mode=="RGBA" and image.width>0 and image.height==64


def test_training_rejects_page_leakage(tmp_path):
    rows=[]
    for i,split in enumerate(["train","validation"]):
        path=tmp_path/f"line_{i}.png";make_crop(path,"a")
        rows.append({"id":str(i),"page_id":"same_page","image":str(path),"text":"a","split":split})
    manifest=tmp_path/"leak.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    with pytest.raises(ValueError,match="more than one data split"):
        train(manifest,tmp_path/"bad.pt",epochs=1,min_lines=1,device_name="cpu")
