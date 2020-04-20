from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import time

curr_time = int(time.time())

NAME = "Cars-classification-xception-sigmoid-100-layers-{}".format(curr_time)

tensorboard = TensorBoard(log_dir='logs\\{}'.format(NAME))

train_dir = 'input/car_data/car_data/train'
validation_dir = 'input/car_data/car_data/test'

input_shape = (299, 299, 3)
target_size = (299, 299)
batch_size = 8
lr = 0.0001
epochs = 30

def build_model(input_shape, num_classes):

    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True

    for _i, _layer in enumerate(base_model.layers):
        if _i < 100:
            _layer.trainable = False
        else:
            _layer.trainable = True

    model = Sequential()
    model.add(base_model)

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))

    return model

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

model = build_model(input_shape, train_generator.num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr),
              metrics=['accuracy'])

model_history = model.fit(
    x=train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // batch_size,
    callbacks=[tensorboard]
)

model.save("xception-{}.h5".format(curr_time))
