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
    frame = cv2.resize(frame, resize_shape)

    if frame is None:
        print("Error: Could not read the image.")
    else:
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

yaml_to_fdi = {
    0: 11,  # Upper right central incisor
    1: 12,  # Upper right lateral incisor
    2: 13,  # Upper right canine
    3: 14,  # Upper right first premolar
    4: 15,  # Upper right second premolar
    5: 16,  # Upper right first molar
    6: 17,  # Upper right second molar
    7: 18,  # Upper right third molar
    8: 21,  # Upper left central incisor
    9: 22,  # Upper left lateral incisor
    10: 23,  # Upper left canine
    11: 24,  # Upper left first premolar
    12: 25,  # Upper left second premolar
    13: 26,  # Upper left first molar
    14: 27,  # Upper left second molar
    15: 28,  # Upper left third molar
    16: 31,  # Lower left central incisor
    17: 32,  # Lower left lateral incisor
    18: 33,  # Lower left canine
    19: 34,  # Lower left first premolar
    20: 35,  # Lower left second premolar
    21: 36,  # Lower left first molar
    22: 37,  # Lower left second molar
    23: 38,  # Lower left third molar
    24: 41,  # Lower right central incisor
    25: 42,  # Lower right lateral incisor
    26: 43,  # Lower right canine
    27: 44,  # Lower right first premolar
    28: 45,  # Lower right second premolar
    29: 46,  # Lower right first molar
    30: 47,  # Lower right second molar
    31: 48,  # Lower right third molar
    32: "Amalgam filling",  # Amalgam filling
    33: "Apical Abscess",  # Apical abscess
    34: "Caries",  # Caries
    35: "Prefabricated post",  # Prefabricated post
    36: "Root canal Obturatien",  # Root canal obturation
    37: "Composite filling",  # Composite filling
    38: "Crown",  # Crown
    39: "Implant",  # Implant
    40: "Object",  # Object
    41: "Residual root"  # Residual root
}
yaml_to_universal = {
    0: "8",  # Upper right central incisor
    1: "7",  # Upper right lateral incisor
    2: "6",  # Upper right canine
    3: "5",  # Upper right first premolar
    4: "4",  # Upper right second premolar
    5: "3",  # Upper right first molar
    6: "2",  # Upper right second molar
    7: "1",  # Upper right third molar
    8: "9",  # Upper left central incisor
    9: "10",  # Upper left lateral incisor
    10: "11",  # Upper left canine
    11: "12",  # Upper left first premolar
    12: "13",  # Upper left second premolar
    13: "14",  # Upper left first molar
    14: "15",  # Upper left second molar
    15: "16",  # Upper left third molar
    16: "24",  # Lower left third molar
    17: "23",  # Lower left second molar
    18: "22",  # Lower left first molar
    19: "21",  # Lower left second premolar
    20: "20",  # Lower left first premolar
    21: "19",  # Lower left canine
    22: "18",  # Lower left lateral incisor
    23: "17",  # Lower left central incisor
    24: "25",  # Lower right central incisor
    25: "26",  # Lower right lateral incisor
    26: "27",  # Lower right canine
    27: "28",  # Lower right first premolar
    28: "29",  # Lower right second premolar
    29: "30",  # Lower right first molar
    30: "31",  # Lower right second molar
    31: "32",  # Lower right third molar
    32: "Amalgam filling",  # Amalgam filling
    33: "Apical Abscess",  # Apical abscess
    34: "Caries",  # Caries
    35: "Prefabricated post",  # Prefabricated post
    36: "Root canal Obturatien",  # Root canal obturation
    37: "Composite filling",  # Composite filling
    38: "Crown",  # Crown
    39: "Implant",  # Implant
    40: "Object",  # Object
    41: "Residual root"  # Residual root
}
fdi_to_universal = {
    11: "8",  # Upper right central incisor
    12: "7",  # Upper right lateral incisor
    13: "6",  # Upper right canine
    14: "5",  # Upper right first premolar
    15: "4",  # Upper right second premolar
    16: "3",  # Upper right first molar
    17: "2",  # Upper right second molar
    18: "1",  # Upper right third molar
    21: "9",  # Upper left central incisor
    22: "10",  # Upper left lateral incisor
    23: "11",  # Upper left canine
    24: "12",  # Upper left first premolar
    25: "13",  # Upper left second premolar
    26: "14",  # Upper left first molar
    27: "15",  # Upper left second molar
    28: "16",  # Upper left third molar
    31: "24",  # Lower left central incisor
    32: "23",  # Lower left lateral incisor
    33: "22",  # Lower left canine
    34: "21",  # Lower left first premolar
    35: "20",  # Lower left second premolar
    36: "19",  # Lower left first molar
    37: "18",  # Lower left second molar
    38: "17",  # Lower left third molar
    41: "25",  # Lower right central incisor
    42: "26",  # Lower right lateral incisor
    43: "27",  # Lower right canine
    44: "28",  # Lower right first premolar
    45: "29",  # Lower right second premolar
    46: "30",  # Lower right first molar
    47: "31",  # Lower right second molar
    48: "32",  # Lower right third molar
    "Amalgam filling": "Amalgam filling",
    "Apical Abscess": "Apical Abscess",
    "Caries": "Caries",
    "Prefabricated post": "Prefabricated post",
    "Root canal Obturatien": "Root canal Obturatien",
    "Composite filling": "Composite filling",
    "Crown": "Crown",
    "Implant": "Implant",
    "Object": "Object",
    "Residual root": "Residual root",
}
conditions = [
    'Amalgam filling',
    'Apical Abscess',###
    'Caries',###
    'Prefabricated post',###
    'Root canal Obturatien',
    'Composite filling',
    'Crown',
    'Implant',
    'Object',
    'Residual root'###
]


# Load a model
# weights_path = './yolov8n.pt'
weights_path = 'runs/segment/train3/weights/best.pt'

model = YOLO(weights_path)
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
