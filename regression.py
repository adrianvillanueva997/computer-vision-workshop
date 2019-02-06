import keras
import numpy as np
import matplotlib.pyplot as plt

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
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dense(1))

model.summary()

model.compile(optimizer="sgd", loss="mse", metrics=['mse'])
model.fit(train_images, train_labels, batch_size=64, epochs=5)

predictions = model.predict(test_images)
print(predictions)
predictions = np.rint(predictions).astype(np.uint8).flatten()
print(predictions)
print(test_labels)

accuracy = (predictions == test_labels).mean()
print(accuracy)


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
