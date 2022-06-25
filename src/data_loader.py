import os

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


def load_data(data_dir='../data/COVID-19_Radiography_Dataset'):
    labels = {'Normal': 0, 'COVID': 1}
    image_path_list, label_list = [], []
    for label, value in labels.items():
        for file in os.listdir(os.path.join(data_dir, label, 'images')):
            image_path_list.append(data_dir + '/{}/images/{}'.format(label, file))
            label_list.append(value)

    image_list = list(map(lambda x: read_image(x), image_path_list))
    return image_list, label_list


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (70, 70)) / 255.0
    return np.asarray(image)


def train_val_test_split(image_list, label_list, ratio=(0.2, 0.2)):
    x = np.asarray(image_list).astype('float32')
    y = to_categorical(label_list, num_classes=2)
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio[0], random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=ratio[1], random_state=42)
    return x_train, x_val, x_test, y_train, y_val, y_test
