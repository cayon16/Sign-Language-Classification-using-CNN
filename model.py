
import os
import cv2
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

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
            labels.append(ord(d) - ord('A'))  
    return np.array(images), np.array(labels)

ROOT_PATH_1 = "C:/Users/ADMIN/Desktop/python code/sign_language/dataa/asl_alphabet_train"
train_data_directory_1 = os.path.join(ROOT_PATH_1, "asl_alphabet_train")
images_1, labels_1 = load_data(train_data_directory_1)

ROOT_PATH_2 = "C:/Users/ADMIN/Desktop/python code/sign_language/dataa"
train_data_directory_2 = os.path.join(ROOT_PATH_2, "train")
images_2, labels_2 = load_data(train_data_directory_2)

ROOT_PATH_3 = "C:/Users/ADMIN/Desktop/python code/sign_language/dataa"
train_data_directory_3 = os.path.join(ROOT_PATH_3, "train_2")
images_3, labels_3 = load_data(train_data_directory_3)

ROOT_PATH_TEST_1 = "C:/Users/ADMIN/Desktop/python code/sign_language/dataa"
test_data_directory_1 = os.path.join(ROOT_PATH_TEST_1, "test")
images_test_1, labels_test_1 = load_data(test_data_directory_1)

# Split train data 2 and 3 into train and test
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(images_2, labels_2, test_size=0.3, random_state=42)
x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(images_3, labels_3, test_size=0.3, random_state=42)

# Combine train sets
x_train = np.concatenate((images_1, x_train_2, x_train_3), axis=0)
y_train = np.concatenate((labels_1, y_train_2, y_train_3), axis=0)

# Combine test sets
x_valid = np.concatenate((x_test_2, x_test_3, images_test_1), axis=0)
y_valid = np.concatenate((y_test_2, y_test_3, labels_test_1), axis=0)

y = y_valid

# test for number of classes on train set 
print("Total number of classes:", len(set(y_train)))
print("Label Array:", [chr(X + ord('A')) for X in set(y_train)])

# test for shape of train and test set 
print("The shape of train set: ", x_train.shape, y_train.shape)
print("The shape of test set: ", x_valid.shape, y_valid.shape)

'''
# show some images in data 
plt.imshow(x_train[0], interpolation='none')
plt.title(f'label: {y_train[0]}')
plt.show()

plt.imshow(x_train[32145], interpolation='none')
plt.title(f'label: {y_train[32145]}')
plt.show()

plt.imshow(x_train[60000], interpolation='none')
plt.title(f'label: {y_train[60000]}')
plt.show()

plt.imshow(x_train[80000], interpolation='none')
plt.title(f'label: {y_train[80000]}')
plt.show()

plt.imshow(x_train[90000], interpolation='none')
plt.title(f'label: {y_train[90000]}')
plt.show()

plt.imshow(x_test[0], interpolation='none')
plt.title(f'label: {y_test[0]}')
plt.show()

plt.imshow(x_test[5000], interpolation='none')
plt.title(f'label: {y_test[5000]}')
plt.show()

plt.imshow(x_test[20000], interpolation='none')
plt.title(f'label: {y_test[20000]}')
plt.show()
'''
# distribution of train and test set 
plt.title('distribution of train set')
sns.countplot(x = y_train)
plt.show()

plt.title('distribution of test set') 
sns.countplot(x = y_valid)
plt.show()



# scaling
x_train = x_train / 255.0
x_test = x_valid / 255.0


# One-hot encoding
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_valid)


# reshape the data before training model
x_train = x_train.reshape(-1, 64, 64, 3)
x_valid = x_valid.reshape(-1, 64, 64, 3)


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20, # độ xoay
    width_shift_range=0.2, # 20% dịch chuyển ngang 
    height_shift_range=0.2, # 20% dọc 
    shear_range=0, # độ cắt góc 
    zoom_range=0, # phóng to, thu nhỏ 20%
    horizontal_flip=True, # lật ngang ảnh ngẫu nhiên 
    fill_mode='nearest' # điền các pixel bị mất do xoay, phóng to, ... bằng pixel gần nhất 
)

datagen.fit(x_train)

print(x_train.size, y_train.size)
print(x_valid.size, y_valid.size)


# MODEL BUILDING 
my_model = Sequential()

# First Convolutional Block
my_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
my_model.add(MaxPool2D(pool_size=(2, 2)))
my_model.add(Dropout(0.25))

# Second Convolutional Block
my_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
my_model.add(MaxPool2D(pool_size=(2, 2)))
my_model.add(Dropout(0.25))

# Third Convolutional Block
my_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
my_model.add(MaxPool2D(pool_size=(2, 2)))
my_model.add(Dropout(0.25))

# Fourth Convolutional Block
my_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
my_model.add(MaxPool2D(pool_size=(2, 2)))
my_model.add(Dropout(0.25))



# Fully Connected Layers
my_model.add(Flatten())
my_model.add(Dense(512, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Dense(24, activation='softmax'))

my_model.summary()



# Compile the model
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
my_model.summary()
my_model.save


# Define checkpoint path
checkpoint_path = "best_model.keras"

# Create ModelCheckpoint callback
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

# use checkpoint to have best val accuracy 
history = my_model.fit(datagen.flow(x_train, y_train, batch_size=512), 
                        epochs=20, 
                        validation_data=(x_valid, y_valid), 
                        shuffle=True,
                        callbacks=[checkpoint])


# load checkpoint and print the information of model 
my_model.load_weights(checkpoint_path)
test_loss, test_accuracy = my_model.evaluate(x_valid, y_valid)
print('MODEL ACCURACY = {}%'.format(test_accuracy * 100))
print('MODEL LOSS = {}%'.format(test_loss))
my_model.save('gray_model.h5')


# Analysis results 
# plot the figure to compare accuracy and val accuracy
epochs = range(1, 21) 
fig, ax = plt.subplots(1, 2, figsize=(16, 9))

# take accuracy from the history of model
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

# plot accuracy figure 
ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
ax[0].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

# plot loss figure
ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
ax[1].plot(epochs, val_loss, 'r-o', label='Validation Loss')
ax[1].set_title('Training & Validation Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")

plt.show()

# Predict on test set 
predictions = my_model.predict(x_test)
y_pred_classes = label_binarizer.inverse_transform(predictions)
y_test_classes = label_binarizer.inverse_transform(y_valid)

# report of model
classes = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(y_test_classes, y_pred_classes, target_names=classes))

# heat map 
cm = confusion_matrix(y_test_classes, y_pred_classes)
cm = pd.DataFrame(cm, index=[i for i in range(25) if i != 9], columns=[i for i in range(25) if i != 9])
plt.figure(figsize=(15, 15))
sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='')
plt.show()
