"""Segmentation evaluation with explicit patient split integrity checks."""
import argparse,hashlib,json
from pathlib import Path

def validate_patient_manifest(rows):
    if not rows:raise ValueError("Patient manifest must not be empty")
    patients={};images=set()
    for row in rows:
        patient=row.get("patient_id");split=row.get("split");image=row.get("image")
        if not isinstance(patient,str) or not patient or split not in ('train','val','test') or not isinstance(image,str) or not image:
            raise ValueError("Each record requires patient_id, split=train/val/test and image")
        if image in images:raise ValueError("Duplicate image in patient manifest")
        if patient in patients and patients[patient]!=split:raise ValueError("Patient appears in multiple splits")
        images.add(image);patients[patient]=split
    if set(patients.values()) != {'train','val','test'}:raise ValueError("All three splits are required")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--patients',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('evaluation.json'))
    args=ap.parse_args()
    rows=json.loads(args.patients.read_text());validate_patient_manifest(rows)
    import yaml
    data=yaml.safe_load(args.data.read_text())
    root=Path(data.get('path',args.data.resolve().parent))
    if not root.is_absolute():root=args.data.resolve().parent/root
    image_extensions={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
    for split in ('train','val','test'):
        paths=data.get(split)
        if not paths:ap.error(f'Missing dataset {split} path')
        entries=paths if isinstance(paths,list) else [paths];actual=set()
        for entry in entries:
            p=Path(entry);p=p if p.is_absolute() else root/p
            if p.is_dir():actual.update(x.resolve() for x in p.rglob('*') if x.suffix.lower() in image_extensions)
            elif p.suffix=='.txt':
                for line in p.read_text().splitlines():
                    if line.strip():
                        image=Path(line.strip());actual.add((image if image.is_absolute() else p.parent/image).resolve())
            else:ap.error('Split must be an image directory or text path list')
        expected={(root/row['image']).resolve() for row in rows if row['split']==split}
        if actual!=expected:ap.error(f'Patient manifest does not exactly cover dataset {split}')
    from ultralytics import YOLO
    result=YOLO(str(args.model)).val(data=str(args.data),split='test')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'metrics':result.results_dict,'patient_manifest_sha256':hashlib.sha256(args.patients.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'split':'test'},indent=2))
if __name__=='__main__':main()
