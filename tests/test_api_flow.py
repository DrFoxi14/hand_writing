from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from hie.dataset.store import DatasetStore


def image_bytes():
    im=Image.new("RGB",(320,240),(248,248,246));d=ImageDraw.Draw(im)
    for x in range(0,320,20):d.line((x,0,x,239),fill=(220,220,220),width=1)
    for y in range(0,240,20):d.line((0,y,319,y),fill=(220,220,220),width=1)
    d.text((30,80),"sample line",fill=(60,70,145))
    bio=BytesIO();im.save(bio,format="PNG");return bio.getvalue()


def test_upload_list_and_line_annotation_api(tmp_path, monkeypatch):
    from hie.api import main
    monkeypatch.setattr(main,"store",DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw"))
    monkeypatch.setattr(main,"PROCESSED",tmp_path/"processed")
    monkeypatch.setattr(main,"ANNOTATIONS",tmp_path/"annotations")
    client=TestClient(main.app)
    response=client.post('/api/pages',files={'file':('sample.png',image_bytes(),'image/png')})
    assert response.status_code==200,response.text
    page=response.json()['page']; page_id=page['id']
    listed=client.get('/api/pages').json()['pages']
    assert len(listed)==1 and listed[0]['id']==page_id
    annotated=client.post('/api/annotations/lines',json={"page_id":page_id,"bbox":[20,60,140,50],"text":"Salut, lume!","split":"validation"})
    assert annotated.status_code==200,annotated.text
    status=client.get('/api/training/status').json()
    assert status['validation_lines']==1
    assert Path(tmp_path/'annotations'/'lines.jsonl').exists()
