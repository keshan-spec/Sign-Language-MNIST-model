import cv2 as cv
import os
import numpy as np
from tensorflow.keras.models import load_model
from data import predict
from termcolor import colored

try:
    model = load_model('models/model.h5')
    print(colored("[MODEL LOADED SUCCESSFULY]", "green"))
except Exception as e:
    print(colored(f"[FAILED TO LOAD MODEL] {e}", "red"))
    exit()

cap = cv.VideoCapture(0)
start = False

try:
    while True:
        _, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame = frame[100:400, 100:400]
        if start:
            region_of_interest = np.array(frame)  # get the frame
            img = cv.cvtColor(region_of_interest, cv.COLOR_RGB2GRAY)
            img = cv.resize(img, (28, 28))
            prediction = predict(model, img)
            print(prediction)
        cv.imshow("Collecting images", frame)
        k = cv.waitKey(10)
        if k == ord('a'): start = not start
        if k == ord('q'): break
except Exception as e:
    print(colored(f"[ERROR] {e}", "red"))
    exit()

cap.release()
cv.destroyAllWindows()
