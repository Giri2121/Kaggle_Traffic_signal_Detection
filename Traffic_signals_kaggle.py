import os
import glob
import shutil

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import plot_confusion_matrix

from PIL.Image import FASTOCTREE

#for the below we have already created the respective functions in the python files so importing them directly.
from my_utils import split_data,order_test_set,create_generators
from deep_learning_models import street_sign_model


if __name__ == '__main__':
    path_to_train = r"C:\Users\vikassaigiridhar\Music\traffic_signal\split_data\Train"
    path_to_val = r"C:\Users\vikassaigiridhar\Music\traffic_signal\split_data\Validation"
    path_to_test = r"C:\Users\vikassaigiridhar\Music\traffic_signal\Test"
    batch_size = 32
    epochs = 15
    lr = 0.001

    train_generator,val_generator,test_generator = create_generators(batch_size,path_to_train,path_to_val,path_to_test)
    no_of_classes = train_generator.num_classes

    TRAIN = False
    TEST = False

    if TRAIN:
        path_to_save_model = './Models'
        check_point_saver = ModelCheckpoint(
            path_to_save_model,
            monitor = "val_accuracy",
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1
        )

        early_stop = EarlyStopping(monitor = 'val_accuracy',patience = 3)
        model = street_sign_model(no_of_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr,amsgrad = True)
        model.compile(optimizer = optimizer,loss = 'categorical_crossentropy',metrics = ['accuracy'])
        model.fit(
            train_generator,
            epochs = epochs,
            batch_size = batch_size,
            validation_data = val_generator,
            callbacks = [check_point_saver,early_stop]
        )

    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print('Evaluating Validation Set')
        model.evaluate(val_generator)

        print('Evaluating Test Set')
        model.evaluate(test_generator)

        plot_confusion_matrix()

