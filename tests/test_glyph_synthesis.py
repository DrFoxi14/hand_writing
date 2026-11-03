from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pytest
from hie.dataset.store import DatasetStore
from hie.synthesis.glyph_bank import GlyphBank, InsufficientGlyphCoverage
from hie.synthesis.engine import GlyphComposer


def add_glyph(store, root, ch, position="any", verified=True, color=(30,70,170,255)):
    path=root/f"{ord(ch)}_{position}_{verified}.png"
    im=Image.new("RGBA",(16,24),(0,0,0,0)); d=ImageDraw.Draw(im); d.line((3,3,8,20),fill=color,width=2); im.save(path)
    return store.add_glyph(ch,position,path,verified)


def test_missing_character_is_reported_not_substituted(tmp_path):
    store=DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw")
    add_glyph(store,tmp_path,"a")
    with pytest.raises(InsufficientGlyphCoverage) as exc: GlyphComposer(GlyphBank(store)).render("ab")
    assert exc.value.missing == ["b"]


def test_only_verified_glyphs_are_used(tmp_path):
    store=DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw")
    add_glyph(store,tmp_path,"a",verified=False)
    assert GlyphBank(store).missing("a") == ["a"]


def test_verified_glyph_composer_renders_newline(tmp_path):
    store=DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw")
    for ch in "abc": add_glyph(store,tmp_path,ch)
    image=GlyphComposer(GlyphBank(store)).render("abc\na",seed=2)
    assert image.mode == "RGBA"
    assert image.height > 24
    assert image.getbbox() is not None

def test_long_text_wraps_at_word_boundaries(tmp_path):
    store=DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw")
    for ch in "abc": add_glyph(store,tmp_path,ch)
    image=GlyphComposer(GlyphBank(store)).render("abc abc",max_width=70,seed=3)
    assert image.width <= 70
    assert image.height > 30


def test_context_specific_glyph_does_not_claim_unseen_positions(tmp_path):
    store=DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw")
    add_glyph(store,tmp_path,"a",position="initial")
    bank=GlyphBank(store)
    report=bank.coverage_report("aaa")
    assert report["missing"] == ["a"]
    assert report["missing_positions"]["a"] == ["final", "medial"]
    with pytest.raises(InsufficientGlyphCoverage):
        GlyphComposer(bank).render("aaa")


def test_any_variant_covers_all_contexts(tmp_path):
    store=DatasetStore(tmp_path/"db.sqlite3",tmp_path/"raw")
    add_glyph(store,tmp_path,"a",position="any")
    assert GlyphBank(store).coverage_report("a aa")["missing"] == []
