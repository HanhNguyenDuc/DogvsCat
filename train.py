import os
import numpy as np
import cv2
from keras.layers import *
from keras.optimizers import *
from keras.datasets import mnist
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input

DIR = 'train'
X = []
y = []
i = 0
test_split_rate = 0.2
val_split_rate = 0.1

# print(os.listdir("../input/train/train"))

IMG_SIZE = (128, 128, 3)
for f in os.listdir(DIR):
    i+=1
    file_path = os.path.join(DIR, f)
    if i % 1000 == 0:
        print('{} images is loaded!!'.format(i))
    if os.path.isfile(file_path):
        img = cv2.imread(file_path, 1)
        resized_img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
        X.append(resized_img)
        label = f.split(".")[0]
        if (label == 'cat'):
            y.append(0)
        else:
            y.append(1)

print('Done, now split to train, test, and validation data !!!')
X_data = np.stack(X, axis = 0)
y_data = np.stack(y, axis = 0)

train_endp = int(X_data.shape[0] * (1 - test_split_rate - val_split_rate))
test_endp = int(X_data.shape[0] * (1 - val_split_rate))

X_train = preprocess_input(X_data[:train_endp])
y_train = y_data[:train_endp]
# print(y_train.shape)
X_test = preprocess_input(X_data[train_endp:test_endp])
y_test = y_data[train_endp:test_endp]
X_val = preprocess_input(X_data[test_endp:])
y_val = y_data[test_endp:]


IMG_SHAPE = X_train.shape[1:]

datagen = ImageDataGenerator(rotation_range=15,
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)


def SiameseNetworkModel():
    input_ = Input(shape = IMG_SHAPE)
    
    #Block conv 1
    conv_1_a = Conv2D(32, (3, 3), padding = 'same')(input_)
    conv_1_b = Conv2D(32, (3, 3), padding = 'same')(conv_1_a)
    bn_1 = BatchNormalization()(conv_1_b)
    maxpool_1 = MaxPool2D((2, 2))(bn_1)
    drop_1 = Dropout(0.25)(maxpool_1)


    #Block conv 2 
    conv_2_a = Conv2D(64, (3, 3), padding = 'same')(drop_1)
    conv_2_b = Conv2D(64, (3, 3), padding = 'same')(conv_2_a)
    bn_2 = BatchNormalization()(conv_2_b)
    maxpool_2 = MaxPool2D((2, 2))(bn_2)
    drop_2 = Dropout(0.25)(maxpool_2)


    #Block conv 3
    conv_3 = Conv2D(128, (3, 3), padding = 'same')(drop_2)
    conv_3_b = Conv2D(128, (3, 3), padding = 'same')(conv_3)
    bn_3 = BatchNormalization()(conv_3_b)
    maxpool_3 = MaxPool2D((2, 2))(bn_3)
    drop_3 = Dropout(0.25)(maxpool_3)

    #Block classification
    flatten_ = Flatten()(drop_3)
    dense_1 = Dense(512)(flatten_)
    batch_norm = BatchNormalization()(dense_1)
    dropout_ = Dropout(0.5)(batch_norm)
    softmax = Dense(1, activation = 'sigmoid')(dropout_)
    
    return Model(inputs = [input_], outputs = [softmax])


model = SiameseNetworkModel()
model.summary()

checkpoint = ModelCheckpoint('weight.hdf5', monitor = 'val_acc', save_best_only = True, mode = 'max')

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), epochs = 200, callbacks = [checkpoint], steps_per_epoch = X_train.shape[0] / 32, verbose = 2, 
                    validation_data = [X_val, y_val])


loss, acc = model.evaluate(X_test, y_test)

print('loss: {}, acc: {}'.format(loss, acc))



#first train 0.825  
#seconnd train 0.807
#third train 0.8592
#fourth train 0.8708
#fifth train 0.9138