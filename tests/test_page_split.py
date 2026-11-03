from pathlib import Path
from PIL import Image
from hie.dataset.store import DatasetStore


def test_line_annotations_keep_pages_in_one_split(tmp_path):
    store=DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw")
    page=store.ingest(Image.new("RGB",(20,20)).tobytes() if False else _png(),"p.png")
    crop=tmp_path/"crop.png";Image.new("L",(5,5),255).save(crop)
    store.add_line_annotation(page['id'],(0,0,5,5),"test",crop,"train")
    try: store.add_line_annotation(page['id'],(0,0,5,5),"another",crop,"validation")
    except ValueError as e: assert "same split" in str(e)
    else: raise AssertionError("mixed page split accepted")

def _png():
    from io import BytesIO
    b=BytesIO();Image.new("RGB",(20,20),(255,255,255)).save(b,format="PNG");return b.getvalue()
