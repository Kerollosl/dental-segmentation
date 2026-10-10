from ultralytics import YOLO
import cv2
import os


def get_key_from_value(value, dict_choice):
    # Returns a key for any given dictionary value
    for key, val in dict_choice.items():
        if val == value:
            return key


def predict_teeth(image_path, tooth_dict, classes, resize_shape=(840, 540), conf=0.5):

    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read the image.")
    else:
        frame = cv2.resize(frame, resize_shape)
        # Predict for only the classes of choice (Tooth numbers or Conditions)
        results = model.predict(frame, classes=classes, conf=conf)

        # Change the segment print labels to be those of the desired tooth numbering system or conditions
        results[0].names = tooth_dict
        annotated_frame = results[0].plot(conf=False, boxes=True, labels=True)  # Plot the predictions on the frame

        # Display the frame with annotations
        cv2.imshow('Image YOLOv8', annotated_frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

""""" There are different systems for numbering teeth. The dictionaries below are used to convert between the
      FDI (World Dental Federation) notation and the Universal numbering system, and from the YAML line indices
      there is also a dictionary for reference to the dental conditions that the model can predict for"""""



from class_mapping import mappings, validate_model_names
yaml_to_fdi, yaml_to_universal = mappings()
fdi_to_universal = {v:yaml_to_universal[k] for k,v in yaml_to_fdi.items()}
conditions = [yaml_to_fdi[i] for i in range(32,42)]

def main():
    global model
    # Load a model
    # weights_path = './yolov8n.pt'
    weights_path = 'runs/segment/train3/weights/best.pt'

    model = YOLO(weights_path)
    validate_model_names(model.names)
    print("Model successfully loaded")

    # YAML to Universal
    tooth_number_indices = [get_key_from_value(get_key_from_value(str(n), fdi_to_universal), yaml_to_fdi) for n in range(1, 33)]

    # YAML to Dental Conditions
    condition_indices = [get_key_from_value(get_key_from_value(n, fdi_to_universal), yaml_to_fdi) for n in conditions]

    # Choose between segmenting tooth numbers or dental conditions
    CHOICE = 0
    prediction_options = [tooth_number_indices, condition_indices]
    pred_choice = prediction_options[CHOICE]
    dir_options = ["./test_images/tooth_number", "./test_images/conditions"]
    dir_choice = dir_options[CHOICE]
    # Predict for test images
    for img in os.listdir(dir_choice):
        predict_teeth(f"{dir_choice}/{img}", yaml_to_universal, pred_choice, resize_shape=(1200, 800), conf=0.25)


if __name__=="__main__":main()
