import json
from pathlib import Path

def mappings():
 rows=json.loads((Path(__file__).parent/"class-manifest.json").read_text())["classes"]
 return ({r["index"]:r["fdi"] for r in rows},{r["index"]:r["universal"] for r in rows})

def validate_model_names(names):
 fdi,_=mappings()
 names=dict(enumerate(names)) if isinstance(names,list) else names
 if set(names)!=set(fdi):raise ValueError("Model class indices differ from manifest")
 # Compare semantic labels, not just number of classes.
 normalize=lambda s: str(s).lower().replace("_"," ").strip()
 for i,label in fdi.items():
  if normalize(names[i])!=normalize(label):
   raise ValueError(f"Model class {i} does not match manifest: {names[i]!r} vs {label!r}")
