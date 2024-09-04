#
#  file:  cifar10_ada_shal.py
#
#  Adadelta optimizer, shallow network on cifar10 dataset
#
#  RTK, 20-Oct-2019 - cifar10_cnn.py
#  Adapted by DOC, 15-Oct-2022 to include model and history.csv output
#
################################################################

import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

batch_size = 64
num_classes = 10
epochs = 60
img_rows, img_cols = 32, 32

x_train = np.load("../data/cifar10/cifar10_train_images.npy")
y_train = np.load("../data/cifar10/cifar10_train_labels.npy")

x_test = np.load("../data/cifar10/cifar10_test_images.npy")
y_test = np.load("../data/cifar10/cifar10_test_labels.npy")

if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)

print("Model parameters = %d" % model.count_params())
print(model.summary())

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test[:1000], y_test[:1000]),
)

score = model.evaluate(x_test[1000:], y_test[1000:], verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


df = pd.DataFrame.from_dict(history.history)
df.to_csv(r"../data/cifar10/cifar10_ada_shal_model.csv", index=False, header=True)

model.save("../data/cifar10/cifar10_ada_shal_model.h5")
