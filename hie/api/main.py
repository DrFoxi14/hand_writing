from __future__ import annotations
import json
import uuid
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
from io import BytesIO
from PIL import Image

from hie.config import ROOT, PROCESSED, ANNOTATIONS, CHECKPOINTS
from hie.dataset.store import DatasetStore
from hie.vision.preprocess import preprocess_file
from hie.vision.profile import analyze_handwriting
from hie.synthesis.glyph_bank import GlyphBank, InsufficientGlyphCoverage
from hie.synthesis.engine import GlyphComposer
from hie.reconstruction.composite import composite_authorized_region, warp_overlay_to_quadrilateral
from hie.training.manifest import export_line_manifest
from hie.evaluation.metrics import evaluate_pair

REPO_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="Handwriting Identity Engine", version="0.1.0",
              description="Local-first analysis and personal handwriting research workbench")
store = DatasetStore()

class LineAnnotation(BaseModel):
    page_id: str
    bbox: list[int] = Field(min_length=4, max_length=4)
    text: str = Field(min_length=1, max_length=1000)
    split: Literal["train", "validation"] = "train"

class GlyphAnnotation(BaseModel):
    page_id: str
    bbox: list[int] = Field(min_length=4, max_length=4)
    character: str = Field(min_length=1, max_length=1)
    position: str = "any"
    verified: bool = False

class GenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    max_width: int = Field(default=1200, ge=64, le=5000)
    line_spacing: float = Field(default=1.65, ge=1.0, le=3.0)
    char_spacing: float = Field(default=0.10, ge=0.0, le=0.8)
    seed: int | None = None

class NeuralGenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    style_annotation_id: str | None = None
    pixels_per_char: int = Field(default=20, ge=8, le=48)

class TrainingRequest(BaseModel):
    epochs: int = Field(default=20, ge=1, le=200)
    min_lines: int = Field(default=40, ge=40, le=5000)


def _png_response(image: Image.Image, filename: str = "handwriting.png"):
    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    return StreamingResponse(stream, media_type="image/png", headers={"Content-Disposition": f'inline; filename="{filename}"'})


@app.get("/api/health")
def health():
    from hie.device import detect_device
    return {"status":"ok","version":"0.1.0","device":detect_device(),"root":str(ROOT)}

@app.post("/api/pages")
async def upload_page(file: UploadFile = File(...)):
    try:
        payload = await file.read()
        page = store.ingest(payload, file.filename or "upload.jpg")
        if page["duplicate"]:
            return {"page":page,"message":"This exact image is already in the dataset; original was not duplicated."}
        out = PROCESSED / page["id"]
        metadata = preprocess_file(page["raw_path"], out)
        store.set_processed(page["id"], out / "rectified.png", metadata)
        page["preprocessing"] = metadata
        return {"page":page,"message":"Original stored unchanged; rectified and ink-mask derivatives created."}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

@app.get("/api/pages")
def list_pages():
    pages = store.list_pages()
    for page in pages:
        try: page["metadata"] = json.loads(page.pop("metadata_json", "{}"))
        except Exception: page["metadata"] = {}
    return {"pages":pages}

@app.get("/api/pages/{page_id}/image")
def page_image(page_id: str, kind: str = "rectified"):
    page = store.get_page(page_id)
    if not page: raise HTTPException(404,"Unknown page")
    if kind == "original": path = Path(page["raw_path"])
    elif kind == "mask": path = PROCESSED / page_id / "ink_mask.png"
    else: path = PROCESSED / page_id / "rectified.png"
    if not path.exists(): raise HTTPException(404,"Image derivative not found")
    return FileResponse(path)

@app.get("/api/profile")
def profile(page_id: str | None = None):
    pages = [store.get_page(page_id)] if page_id else store.list_pages()[:3]
    pages = [p for p in pages if p]
    if not pages: raise HTTPException(404,"Upload a page first")
    reports = []
    for page in pages:
        img_path = PROCESSED / page["id"] / "rectified.png"
        mask_path = PROCESSED / page["id"] / "ink_mask.png"
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is not None and mask is not None:
            reports.append({"page_id":page["id"],**analyze_handwriting(image,mask)})
    return {"profile":"initial_visual_profile","sample_count":len(reports),"pages":reports,
            "model_ready":False,"note":"Measurements are provisional; annotate line/glyph samples before training or arbitrary-text generation."}

@app.post("/api/annotations/lines")
def annotate_line(payload: LineAnnotation):
    page = store.get_page(payload.page_id)
    if not page: raise HTTPException(404,"Unknown page")
    image_path = PROCESSED / payload.page_id / "rectified.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None: raise HTTPException(404,"Rectified page unavailable")
    x,y,w,h = payload.bbox
    if min(x,y,w,h) < 0 or w < 2 or h < 2 or x+w > image.shape[1] or y+h > image.shape[0]:
        raise HTTPException(400,"Bounding box lies outside the rectified page")
    ink_mask = cv2.imread(str(PROCESSED / payload.page_id / "ink_mask.png"), cv2.IMREAD_GRAYSCALE)
    if ink_mask is None: raise HTTPException(404,"Ink mask unavailable")
    # White paper / black ink is the target representation used by the neural trainer.
    crop = 255 - ink_mask[y:y+h, x:x+w]
    crop_dir = PROCESSED / payload.page_id / "lines"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{uuid.uuid4().hex}.png"
    cv2.imwrite(str(crop_path), crop)
    try:
        annotation = store.add_line_annotation(payload.page_id, tuple(payload.bbox), payload.text,
                                               crop_path, payload.split)
        export_line_manifest(store, ANNOTATIONS / "lines.jsonl")
        return {"annotation":annotation,"manifest":"dataset/annotations/lines.jsonl"}
    except Exception as exc:
        crop_path.unlink(missing_ok=True)
        raise HTTPException(400,str(exc)) from exc

@app.post("/api/annotations/glyphs")
def annotate_glyph(payload: GlyphAnnotation):
    page = store.get_page(payload.page_id)
    if not page: raise HTTPException(404,"Unknown page")
    img = cv2.imread(str(PROCESSED / payload.page_id / "rectified.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(PROCESSED / payload.page_id / "ink_mask.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None: raise HTTPException(404,"Page derivatives unavailable")
    x,y,w,h = payload.bbox
    if min(x,y,w,h) < 0 or w < 2 or h < 2 or x+w > img.shape[1] or y+h > img.shape[0]:
        raise HTTPException(400,"Bounding box lies outside the rectified page")
    crop = img[y:y+h,x:x+w]
    alpha = mask[y:y+h,x:x+w]
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)
    rgba[:,:,3] = alpha
    glyph_dir = ANNOTATIONS / "glyphs"
    glyph_dir.mkdir(parents=True,exist_ok=True)
    glyph_path = glyph_dir / f"{uuid.uuid4().hex}.png"
    Image.fromarray(rgba).save(glyph_path)
    try:
        record = store.add_glyph(payload.character,payload.position,glyph_path,payload.verified,payload.page_id)
        return {"glyph":record,"message":"Glyph stored. It remains unavailable to synthesis until manually verified." if not payload.verified else "Verified glyph added."}
    except Exception as exc:
        glyph_path.unlink(missing_ok=True)
        raise HTTPException(400,str(exc)) from exc

@app.get("/api/coverage")
def coverage(text: str):
    return GlyphBank(store).coverage_report(text)

@app.post("/api/generate")
def generate(payload: GenerateRequest):
    bank = GlyphBank(store)
    report = bank.coverage_report(payload.text)
    if report["missing"]:
        raise HTTPException(422,detail={"message":"Insufficient verified handwriting coverage; no synthetic fallback will be used.",**report})
    try:
        image = GlyphComposer(bank).render(payload.text,payload.max_width,payload.line_spacing,
                                           payload.char_spacing,payload.seed)
        return _png_response(image)
    except InsufficientGlyphCoverage as exc:
        raise HTTPException(422,detail={"message":str(exc),"missing":exc.missing}) from exc
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc

@app.post("/api/generate/neural")
def generate_neural(payload: NeuralGenerateRequest):
    rows = store.list_line_annotations()
    row = next((r for r in rows if r["id"] == payload.style_annotation_id), None) if payload.style_annotation_id else next((r for r in rows if r["split"] == "train"), None)
    if not row:
        raise HTTPException(422,"Add a line annotation and select it as a style reference first")
    try:
        from hie.inference.neural import generate_from_checkpoint
        image = generate_from_checkpoint(payload.text,row["crop_path"],CHECKPOINTS / "writer_v1.pt",pixels_per_char=payload.pixels_per_char)
        return _png_response(image,"neural-handwriting.png")
    except FileNotFoundError as exc:
        raise HTTPException(404,str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422,str(exc)) from exc

@app.post("/api/reconstruct")
async def reconstruct(original: UploadFile = File(...), overlay: UploadFile = File(...),
                      polygon: str = Form(...)):
    try:
        poly = json.loads(polygon)
        original_image = np.asarray(Image.open(BytesIO(await original.read())).convert("RGB"))
        overlay_image = np.asarray(Image.open(BytesIO(await overlay.read())).convert("RGBA"))
        overlay_image = warp_overlay_to_quadrilateral(overlay_image,poly)
        result = composite_authorized_region(original_image,overlay_image,poly)
        return _png_response(Image.fromarray(result),"notebook-edited.png")
    except Exception as exc:
        raise HTTPException(400,str(exc)) from exc

@app.post("/api/evaluate")
async def evaluate(reference: UploadFile = File(...), generated: UploadFile = File(...),
                   reference_text: str = Form(""), generated_text: str = Form("")):
    try:
        ref = np.asarray(Image.open(BytesIO(await reference.read())).convert("RGB"))
        gen = np.asarray(Image.open(BytesIO(await generated.read())).convert("RGBA" if (generated.content_type or "").endswith("png") else "RGB"))
        return evaluate_pair(ref,gen,reference_text or None,generated_text or None)
    except Exception as exc:
        raise HTTPException(400,str(exc)) from exc

@app.get("/api/training/status")
def training_status():
    lines = store.list_line_annotations()
    manifest = export_line_manifest(store, ANNOTATIONS / "lines.jsonl")
    chars = sorted({c for row in lines if row["split"] == "train" for c in row["text"]})
    return {"annotated_lines":len(lines),"train_lines":sum(r["split"]=="train" for r in lines),
            "validation_lines":sum(r["split"]=="validation" for r in lines),
            "unique_characters":len(chars),"characters":chars,
            "minimum_recommended_lines":40,"ready":sum(r["split"]=="train" for r in lines)>=40 and sum(r["split"]=="validation" for r in lines)>=1,
            "line_annotations":[{"id":r["id"],"page_id":r["page_id"],"text":r["text"],"split":r["split"]} for r in lines],
            "manifest":str(manifest),"checkpoint_exists":(CHECKPOINTS / "writer_v1.pt").exists()}

@app.post("/api/training/run")
def run_training(payload: TrainingRequest):
    try:
        from hie.training.train import train
        result = train(ANNOTATIONS / "lines.jsonl",CHECKPOINTS / "writer_v1.pt",
                       epochs=payload.epochs,min_lines=payload.min_lines)
        return result
    except Exception as exc:
        raise HTTPException(422,str(exc)) from exc

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(REPO_ROOT / "app/frontend/index.html")

app.mount("/static", StaticFiles(directory=REPO_ROOT / "app/frontend"), name="static")
