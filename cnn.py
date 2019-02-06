import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
(train_images, train_labels), (test_images,
                               test_labels) = keras.datasets.mnist.load_data()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])

# plt.show()

train_images = train_images/255
test_images = test_images/255
keras.backend.clear_session()

train_one_hot_labels = to_categorical(train_labels)
test_one_hot_labels = to_categorical(test_labels)
print(test_one_hot_labels)
train_shape = np.shape(train_one_hot_labels)

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=2,
                              padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=32, kernel_size=2,
                              padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer="sgd",
              loss="categorical_crossentropy", metrics=['accuracy'])

reshaped_train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

model.fit(reshaped_train_images, train_one_hot_labels,
          batch_size=32, epochs=3)

predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=1)


def plot_images(i, predictions, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i], cmap=plt.cm.binary)
    color = 'blue' if predictions[i] == true_label[i] else 'red'
    plt.xlabel('{}{}'.format(predictions[i], true_label[i]), color=color)


num_rows, num_cols = 4, 4
num_images = num_rows * num_cols
plt.figure(figsize=(8, 8))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plot_images(i, predictions, test_labels, test_images)
plt.show()
