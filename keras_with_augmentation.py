import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.utils import plot_model
from keras import optimizers
import matplotlib.image as mpimg
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def build_model1():
	model = Sequential()
	model.add(Conv2D(32,(3,3), activation='relu', padding='same', input_shape=(28,28,1,)))
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Flatten())
	model.add(Dense(10, activation='softmax'))
	return model

def build_model2():
	model = Sequential()
	model.add(Conv2D(32,(3,3), activation='relu', padding='same', input_shape=(28,28,1,)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(32,(3,3), activation='relu', padding='same') )
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	return model


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print "X_train shape", X_train.shape
print "y_train shape", y_train.shape
print "X_test shape", X_test.shape
print "y_test shape", y_test.shape

# reshape
X_train = X_train.reshape( X_train.shape[0], X_train.shape[1], X_train.shape[2], 1 )
X_test = X_test.reshape( X_test.shape[0], X_test.shape[1], X_test.shape[2], 1 )
#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)

# normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print "Training set shape", X_train.shape
print "Testing set shape", X_test.shape

# convert labels to categories
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# build the model
model = build_model2()

# compile the model
model.summary()
optimizer = optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

# --- image augmentation
aug_rot = 15 #augmentation rotation in degrees
aug_shift = 0.1 #augmentation shift
aug_shear = 0.1 # image shear augmentation
aug_zoom = 0.1 # image zoom augmentation

datagen = ImageDataGenerator(rotation_range=aug_rot, 
                             width_shift_range=aug_shift, 
                             height_shift_range=aug_shift,
                             zoom_range = aug_zoom,
                             shear_range = aug_shear)

#--- instead of model.fit, use model.fit_generator, with datagen.flow(...) to generate the data per batch!
#history = model.fit([X_train,], Y_train, batch_size=32, nb_epoch=50, validation_split=0.33, verbose=1)
batch_size = 32
epochs = 100

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), 
							  steps_per_epoch=40000/batch_size,
						      validation_data=datagen.flow(X_train, Y_train, batch_size=batch_size),
                    		  validation_steps=20000/batch_size, 
                    		  epochs=epochs)

score = model.evaluate(X_test, Y_test, verbose=0)
print 'Test score:', score

model_name = "CNN_model2aug"

with open("output/"+model_name+"_test_score.txt", 'w') as f:
    print >> f, score

# save history to csv file
pd.DataFrame(history.history).to_csv("output/"+model_name+"_history.csv")