from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
from hie.vision.preprocess import preprocess_array
from hie.vision.profile import analyze_handwriting


def main():
    parser=argparse.ArgumentParser(description="Analyze a set of local handwriting reference photos")
    parser.add_argument("images",nargs="+",type=Path)
    parser.add_argument("--out",type=Path,default=Path("dataset/evaluation/reference-analysis"))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for index,path in enumerate(args.images,1):
        image=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if image is None: raise SystemExit(f"Cannot read image: {path}")
        result=preprocess_array(image)
        folder=args.out/f"page_{index}";folder.mkdir(exist_ok=True)
        cv2.imwrite(str(folder/"rectified.png"),result.image)
        cv2.imwrite(str(folder/"ink_mask.png"),result.ink_mask)
        reports.append({"file":path.name,"source_size":[image.shape[1],image.shape[0]],
          "geometry":{"detected":result.geometry.detected,"method":result.geometry.method,
                      "corners":result.geometry.corners},"grid_spacing":result.grid_spacing,
          "line_box_proposals":result.line_boxes,"profile":analyze_handwriting(result.image,result.ink_mask)})
    out=args.out/"report.json";out.write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding="utf-8")
    print(out)

if __name__=="__main__":main()
