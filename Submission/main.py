import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
import imutils
from PIL import Image
import numpy as np
import time

from keypoint_model import keypoint_model
import emotion_model as emotion_model
from utils import *

device = 'cuda:0' if torch.cuda_is_available() else 'cpu'

model = keypoint_model()
model.load_state_dict(torch.load("./models/keypoint_detection_model.pth", map_location='cpu'))
model.eval()
classifier = get_classifier(data_dir="./models", file_name="face_detection_model.xml")

dog_filter = cv2.imread("filters/dog.png")
dog_filter_alt = cv2.imread("filters/dog_2.png")
eyes = get_eyes_filter()
sunglasses_filter = cv2.imread("filters/sunglasses.png", cv2.IMREAD_UNCHANGED)
flag_filter = cv2.imread("filters/flag.png", cv2.IMREAD_UNCHANGED)

net = emotion_model.Net().float().to(device)
emotion_model_path = './models/saved_models/best_emotion_model.pt'
net.load_state_dict(torch.load(emotion_model_path, map_location='cpu'))


def main():
    cap = cv2.VideoCapture(0)
    while(True):
        emotion_detected = 0
        ret, frame = cap.read()
        faces = detect_faces(classifier=classifier, image=frame)
        images, keypoints = detect_keypoints(faces=faces, image=frame, model=model, padding=50)
        if len(images) <= 0:
            continue
        x, y, w, h = faces[0]
        emotion_detected = emotion_predictor(
            emotion_classifier, frame[faces[0][0]: faces[0][0]+faces[0][2], faces[0][1]:faces[0][1] + faces[0][3]])
        print('Emotion Detected: ', emotion_detected)
        if emotion_detected == 3:  # happy
            filter_image = apply_sunglasses_filter(
                frame.copy(), sunglasses_filter, eyes, 0, h/3, 0, w, x, y, w, h)
        elif emotion_detected == 4:  # sad
            filter_image = dog_face_filter(frame, classifier, dog_filter)
        elif emotion_detected == 5:  # surprised
            filter_image = apply_flag_filter(frame, flag_filter, keypoints, faces[0])
        else:
            filter_image = dog_face_filter(frame, classifier, dog_filter_alt)
        cv2.putText(filter_image, str(EMOTIONS[emotion_detected]), (int(
            filter_image.shape[1]/2), int(filter_image.shape[0]/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        time.sleep(0.3)
        cv2.imshow('frame', filter_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
