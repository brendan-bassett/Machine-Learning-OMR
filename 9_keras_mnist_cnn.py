# USAGE
# python keras_mnist.py --output output/keras_mnist.png

# import the necessary packages
#from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import tf.keras.utils.np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist


batch_size = 128
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape(trainX.shape[0], 28, 28, 1).astype('float')/255.0 # flatten training images
testX = testX.reshape(testX.shape[0], 28, 28, 1).astype('float')/255.0 # flatten test images


# convert the labels from integers to vectors
trainY = keras.utils.to_categorical(trainY, num_classes=num_classes)
testY = keras.utils.to_categorical(testY, num_classes=num_classes)


## define the 784-256-128-10 architecture using Keras
#model = Sequential()
#model.add(Dense(256, input_shape=(784,), activation="relu"))  #'sigmoid'
#model.add(Dense(128, activation="relu"))
#model.add(Dense(10, activation="softmax"))
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), padding="same",
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))




# train the model usign SGD
print("[INFO] training network...")
sgd = SGD(lr= 0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=5, batch_size=128)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1)))
	#target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 5), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 5), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 5), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 5), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
#plt.savefig(args["output"])