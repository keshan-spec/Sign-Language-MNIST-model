import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data import Data # gathered data from csv


obj = Data()
SIZE = obj.SIZE # size of the image (28, 28)
CLASSES = obj.LABELS # all the classes (a,b,...,z)

# DATA
test_images, test_labels = obj.load_with_np('data/sign_mnist_test.csv')  # get test data
test_images = test_images / 255.0

try:
    model = load_model('models/model.h5')
except Exception as e:
    print(e)
    exit()