# YOLOv8 Tooth Number and Dental Condition Model

A YOLOv8 segmentation model for dental X-rays that can either label each tooth with its Universal numbering-system number (1–32, including wisdom teeth) or flag visible dental conditions. Switch modes via a single flag in `test.py`.

### Model Architecture
- YOLOv8

### Necessary Packages
- ultralytics: 8.2.76
- cv2: 4.8.0.76
- roboflow: 1.1.37
- yaml: 6.0.2
- IPython: 7.34.0

### Directions

1. In a Python notebook environment, upload the `/runs/segment/train3/weights/best.pt` file to allow the model to begin training with pre-trained weights. Run the `train_and_validate.ipynb` notebook. Note: This process is done in a notebook to leverage simplified downloading of the Roboflow dataset and use of a virtual GPU.
2. Once the notebook has been run, download the created `runs.zip` file. Unzip and upload to the same directory as the `test.py` script. Navigate in the runs directory to find the path for the new `best.pt` file. Change the path in the `test.py` script on line 185 to match this path. 
3. Run the `test.py` script. This runs the model on several test cases and segments either the universal numbering tooth number for each tooth or the dental conditions spotted in the x-ray image. The default setting is predicting the dental conditions. To navigate between the two segmentation tasks, change line 197 in `test.py` to be 0 for tooth number prediction or 1 for dental condition prediction.
4. To add more test cases, upload an image to the `test_images/` subdirectory. The image should be an x-ray of ones teeth. The model works best on front facing, non mirrored X-rays of the entire mouth. The model is able to segment wisdom teeth (1, 16, 17, and 32).

### Contents

- `test.py` - Program to load model and test it with test cases
- `train_and_validate.ipynb` - Program to download Roboflow dataset, load, train, and validate YOLOv8 model, and save weights and validation metrics to a zip file to be downloaded. 
- `/runs/segment` - Subdirectory of YOLOv8 model training and validation containing the training weights and validation metrics.
- `/runs/segment/train3/weights/best.pt` - Highest performing model weights file from previous trainings
- `FDI_Numbering_System.png` - Image showing the FDI system of numbering teeth. The Roboflow dataset used labels the teeth with this naming convention, however it is not as highly used as the Universal Numbering system. 
- `Universal_Numbering_System.png` - Image showing the Universal system of numbering teeth. `test.py` uses a dictionary to convert the labels when displaying segmentations from the FDI system to this Universal system. 
- `/test_images/` - Subdirectory containing images to test the model's performance.

## Cleanup behavior and validation limits

Research-only tooth segmentation. Unreadable images are rejected before resizing. Set ROBOFLOW_API_KEY externally and revoke the previously committed credential. Tooth/condition mappings and patient-separated evaluation still require validation against the exact model.names and original dataset provenance.

Use `python predict.py INPUT --model WEIGHTS --output predictions` for noninteractive image inference. Unreadable images fail before inference.

For explicit held-out evaluation, supply a JSON array of `{patient_id, split, image}` records (image paths relative to dataset root) and run `python evaluate.py --model WEIGHTS --data data.yaml --patients patients.json --output evaluation.json`. The tool rejects patient leakage and mismatched image coverage, then evaluates only test and records checksums. Patient identities/provenance are supplied by the dataset owner; they are not inferred from filenames.
