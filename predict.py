"""Research-only segmentation CLI; full accuracy requires independent evaluation."""
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input",type=Path)
    ap.add_argument("--model",type=Path,default=Path("runs/segment/train3/weights/best.pt"))
    ap.add_argument("--output",type=Path,default=Path("predictions"))
    ap.add_argument("--confidence",type=float,default=.25)
    args=ap.parse_args()
    if not args.input.is_file():ap.error("input file does not exist")
    if not args.model.is_file():ap.error("model weights do not exist")
    if not 0 <= args.confidence <= 1:ap.error("confidence must be between 0 and 1")
    import cv2
    image=cv2.imread(str(args.input))
    if image is None:ap.error("input is not a readable image")
    from ultralytics import YOLO
    model=YOLO(str(args.model))
    results=model.predict(image,conf=args.confidence,save=False)
    args.output.mkdir(parents=True,exist_ok=True)
    for i,result in enumerate(results):
        if not cv2.imwrite(str(args.output/f"prediction-{i}.png"),result.plot()):raise OSError("could not save annotated image")
    (args.output/"metadata.json").write_text(json.dumps({"input":str(args.input),"model":str(args.model),"classes":model.names,"confidence":args.confidence},indent=2))
if __name__=="__main__":main()
