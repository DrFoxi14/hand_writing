from pathlib import Path
from PIL import Image
from io import BytesIO
from hie.dataset.store import DatasetStore


def jpeg_bytes():
    bio=BytesIO(); Image.new("RGB",(24,32),(255,255,255)).save(bio,format="JPEG"); return bio.getvalue()


def test_ingest_is_append_only_and_deduplicates(tmp_path):
    store=DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw")
    data=jpeg_bytes()
    first=store.ingest(data,"page.jpg")
    second=store.ingest(data,"copy.jpg")
    assert first["id"] == second["id"]
    assert second["duplicate"] is True
    assert Path(first["raw_path"]).read_bytes() == data
    assert len(store.list_pages()) == 1


def test_unsupported_extension_is_rejected(tmp_path):
    store=DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw")
    try: store.ingest(b"not an image","page.txt")
    except ValueError as e: assert "Unsupported" in str(e)
    else: raise AssertionError("unsupported extension accepted")


def test_line_split_is_restricted_to_train_or_validation(tmp_path):
    from pathlib import Path
    import pytest
    store=DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw")
    page=store.ingest(jpeg_bytes(),"page.jpg")
    with pytest.raises(ValueError,match="split must"):
        store.add_line_annotation(page["id"],(0,0,10,10),"test",Path(tmp_path/"crop.png"),"test")
