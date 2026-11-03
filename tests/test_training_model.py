import pytest

torch=pytest.importorskip("torch")
from hie.training.model import WriterConditionedGenerator

def test_writer_conditioned_model_output_shape():
    model=WriterConditionedGenerator(vocab_size=12)
    chars=torch.tensor([[1,2,3,4]])
    style=torch.rand(1,1,64,256)
    out=model(chars,style,output_height=64,pixels_per_char=20)
    assert tuple(out.shape)==(1,1,64,80)
    assert torch.isfinite(out).all()
