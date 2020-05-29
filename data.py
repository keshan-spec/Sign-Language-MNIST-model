import numpy as np

class Data:
    def __init__(self):
        self.SIZE = (28,28)
        self.LABELS = dict(zip(range(0, 25), list(map(chr, range(97, 123)))))
    
    def load_with_file(self, path):
        with open(path, 'r') as f:
            _, pixels = f.readline(), f.readlines()
            np_pixels = np.array([pixel.split(',') for pixel in pixels])

        labels, data = np_pixels[0:, 0].astype(np.uint8), np.array([np.reshape(i[1:], self.SIZE) for i in np_pixels.astype(np.uint8)])
        return data, labels

    def load_with_np(self, path):
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        labels = data[0:, 0].astype(np.uint8)
        data = np.array([np.reshape(i[1:], self.SIZE) for i in data.astype(np.uint8)])

        return data, labels
