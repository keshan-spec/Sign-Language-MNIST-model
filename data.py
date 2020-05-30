from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os


class Data:
    def __init__(self):
        self.SIZE = (28, 28)
        self.LABELS = dict(zip(range(0, 25), list(map(chr, range(97, 123)))))

    def load_with_file(self, path):
        with open(path, 'r') as f:
            _, pixels = f.readline(), f.readlines()
            np_pixels = np.array([pixel.split(',') for pixel in pixels])

        labels, data = np_pixels[0:, 0].astype(np.uint8), np.array(
            [np.reshape(i[1:], self.SIZE) for i in np_pixels.astype(np.uint8)])
        return data, labels

    def load_with_np(self, path):
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        labels = data[0:, 0].astype(np.uint8)
        data = np.array([np.reshape(i[1:], self.SIZE) for i in data.astype(np.uint8)])

        return data, labels


# check the accuracy of the model
def predict(model, img, verbose=False):
    obj = Data()
    CLASSES = obj.LABELS
    prediction = model.predict(np.array([img]))
    return CLASSES[np.argmax(prediction)]


# generates transformed images for more data
def generate_images(img, prefix='sample'):
    if not os.path.exists("generated_images"): os.mkdir("generated_images")
    # creates a data generator object that transforms images
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    try:
        # pick an image to transform
        img = np.array(image.img_to_array(img))  # convert image to numpy array
        img = img.reshape((1,) + img.shape)  # reshape image
        i = 0
        # this loops runs forever until we break, saving images to current directory with specified prefix
        for batch in datagen.flow(img, save_to_dir="generated_images", save_prefix='test', save_format='jpeg'):
            plt.figure(i)
            plt.imshow(image.img_to_array(batch[0]))
            cv.imwrite(f'{prefix}_{i}.jpg', image.img_to_array(batch[0]))
            i += 1
            if i > 4: break
        plt.show()
    except Exception as e:
        print(f"[ERROR]  {e}")
