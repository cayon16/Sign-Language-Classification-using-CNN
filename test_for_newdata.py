
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory)]
        for f in file_names:
            img = cv2.imread(f)
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(ord(d) - ord('A'))  # Chuyển đổi ký tự thành số từ 0-25
    return np.array(images), np.array(labels)

ROOT_PATH_TEST = "C:/Users/ADMIN/Desktop/python code/sign_language/dataa"
test_data_directory = os.path.join(ROOT_PATH_TEST, "test")
images_test, labels_test = load_data(test_data_directory)

x = np.array(images_test)
y = np.array(labels_test)

x.reshape(-1,64,64,3)
x = x/255.0

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_hot = label_binarizer.fit_transform(y)
# y = label_binarizer.fit_transform(y)




model = tf.keras.models.load_model('new_model.h5')
predictions = model.predict(x)
y_pred = label_binarizer.inverse_transform(predictions)

count = 0
for i in range(len(y_pred)):
    if y_pred[i] == y[i]:
        count += 1

print('Accuracy: ', (count/len(y_pred))*100, '%')
