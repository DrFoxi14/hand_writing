from __future__ import annotations
import re
import cv2
import numpy as np


def _edit_distance(a: list, b: list) -> int:
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    return _edit_distance(list(reference),list(hypothesis))/max(1,len(reference))


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref=re.findall(r"\S+",reference,flags=re.UNICODE)
    hyp=re.findall(r"\S+",hypothesis,flags=re.UNICODE)
    return _edit_distance(ref,hyp)/max(1,len(ref))


def _gray_pair(reference: np.ndarray, generated: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    def gray(im):
        if im.ndim==3 and im.shape[2]==4:
            rgb=im[:,:,:3]
            alpha=im[:,:,3:4].astype(np.float32)/255
            white=np.full_like(rgb,255)
            rgb=(rgb*alpha+white*(1-alpha)).astype(np.uint8)
        else: rgb=im
        return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) if rgb.ndim==3 else rgb
    a,b=gray(reference),gray(generated)
    # Align dimensions by resizing generated to the reference crop for comparable image metrics.
    b=cv2.resize(b,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_AREA)
    return a.astype(np.float32)/255.0,b.astype(np.float32)/255.0


def structural_similarity(reference: np.ndarray, generated: np.ndarray) -> float:
    a,b=_gray_pair(reference,generated)
    mu_a=cv2.GaussianBlur(a,(11,11),1.5); mu_b=cv2.GaussianBlur(b,(11,11),1.5)
    sigma_a=cv2.GaussianBlur(a*a,(11,11),1.5)-mu_a*mu_a
    sigma_b=cv2.GaussianBlur(b*b,(11,11),1.5)-mu_b*mu_b
    sigma_ab=cv2.GaussianBlur(a*b,(11,11),1.5)-mu_a*mu_b
    c1=0.01**2; c2=0.03**2
    ssim=((2*mu_a*mu_b+c1)*(2*sigma_ab+c2))/((mu_a*mu_a+mu_b*mu_b+c1)*(sigma_a+sigma_b+c2)+1e-12)
    return float(np.clip(np.mean(ssim),-1,1))


def ink_iou(reference: np.ndarray, generated: np.ndarray) -> float:
    a,b=_gray_pair(reference,generated)
    ma=a<0.72; mb=b<0.72
    union=np.logical_or(ma,mb).sum()
    return float(np.logical_and(ma,mb).sum()/union) if union else 1.0


def foreground_width_proxy(image: np.ndarray) -> float | None:
    if image.ndim==3 and image.shape[2]==4:
        rgb=image[:,:,:3].astype(np.float32)
        alpha=image[:,:,3:4].astype(np.float32)/255.0
        rgb=(rgb*alpha+255.0*(1.0-alpha)).astype(np.uint8)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    else:
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) if image.ndim==3 else image
    fg=(gray<180).astype(np.uint8)
    if not fg.any(): return None
    dist=cv2.distanceTransform(fg,cv2.DIST_L2,3)
    vals=dist[fg>0]
    return float(2*np.median(vals)) if len(vals) else None


def evaluate_pair(reference: np.ndarray, generated: np.ndarray, reference_text: str | None = None,
                  generated_text: str | None = None) -> dict:
    a,b=_gray_pair(reference,generated)
    result={"ssim":structural_similarity(reference,generated),"ink_iou":ink_iou(reference,generated),
            "foreground_width_proxy_reference_px":foreground_width_proxy(reference),
            "foreground_width_proxy_generated_px":foreground_width_proxy(generated),
            "mae_gray":float(np.mean(np.abs(a-b))),
            "warning":"Image metrics require aligned crops and are not a standalone proof of writer identity."}
    if reference_text is not None and generated_text is not None:
        result["cer"]=character_error_rate(reference_text,generated_text)
        result["wer"]=word_error_rate(reference_text,generated_text)
        result["text_exact_match"]=reference_text==generated_text
    return result
