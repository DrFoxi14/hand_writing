import numpy as np
from hie.evaluation.metrics import character_error_rate, word_error_rate, evaluate_pair


def test_cer_and_wer_are_unicode_aware():
    assert character_error_rate("țară", "țară") == 0
    assert character_error_rate("țară", "tara") > 0
    assert word_error_rate("Acesta este un exemplu", "Acesta este un exemplu") == 0


def test_image_metrics_identical_pair():
    image=np.full((64,160,3),255,dtype=np.uint8)
    image[20:44,30:40]=[30,50,130]
    metrics=evaluate_pair(image,image,"Salut","Salut")
    assert metrics["ssim"] > 0.99
    assert metrics["ink_iou"] == 1.0
    assert metrics["cer"] == 0
    assert metrics["text_exact_match"] is True


def test_rgba_generated_image_is_supported_by_width_proxy():
    from hie.evaluation.metrics import foreground_width_proxy
    rgba=np.zeros((64,120,4),dtype=np.uint8)
    rgba[15:48,35:44,:3]=[20,40,110]
    rgba[15:48,35:44,3]=255
    assert foreground_width_proxy(rgba) is not None
