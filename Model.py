import math
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import (MaxPooling3D, Conv3D, BatchNormalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2


def createplot(datatrain, dataval, figurename, name, typeofdata):
    # print(datatrain)
    plt.figure(figurename)
    plt.plot(datatrain)
    plt.plot(dataval)
    plt.title(figurename)
    plt.ylabel(typeofdata)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(name + ".png")
    plt.cla()
    plt.clf()
    plt.close()


class Model:
    def __init__(self, batch, weightDecay, stepsPerEpoch, nbEpoch):
        self.batch = batch
        self.weightDecay = weightDecay
        self.stepsPerEpoch = stepsPerEpoch
        self.nbEpoch = nbEpoch

    def Convolutional3DFunctional(self, classes, inputShape, weightDecay):
        imageInputs = Input(inputShape)
        # FIRST LAYER convolutional3D1
        convolutional3D1 = Conv3D(64, (3, 3, 3),
                                  activation='relu',
                                  strides=(1, 1, 1),
                                  padding='same',
                                  kernel_constraint=l2(weightDecay),
                                  use_bias=False,
                                  name='Convolutional3D1')(imageInputs)
        convolutional3D1 = BatchNormalization(name='BatchNormalization1')(convolutional3D1)

        convolutional3D1 = MaxPooling3D(pool_size=(1, 2, 2),
                                        strides=(1, 2, 2),
                                        padding='valid',
                                        name='pool1')(convolutional3D1)
        # SECOND LAYER convolutional3D2
        convolutional3D2 = Conv3D(128, (3, 3, 3),
                                  activation='relu',
                                  strides=(1, 1, 1),
                                  padding='same',
                                  kernel_constraint=l2(weightDecay),
                                  use_bias=False,
                                  name='Convolutional3D1')(convolutional3D1)
        convolutional3D2 = BatchNormalization(name='BatchNormalization1')(convolutional3D2)
        convolutional3D2 = MaxPooling3D(pool_size=(1, 2, 2),
                                        strides=(1, 2, 2),
                                        padding='valid',
                                        name='pool1')(convolutional3D2)
        # THIRD LAYER convolutional3D3
        convolutional3D3 = Conv3D(256, (3, 3, 3),
                                  activation='relu',
                                  strides=(1, 1, 1),
                                  padding='same',
                                  kernel_regularizer=l2(weightDecay),
                                  use_bias='False',
                                  name='convolutional3D3')(convolutional3D2)
        # FOURTH LAYER convolutional3D4
        convolutional3D4 = Conv3D(256, (3, 3, 3),
                                  activation='relu',
                                  strides=(1, 1, 1),
                                  padding='same',
                                  kernel_regularizer=l2(weightDecay),
                                  use_bias='False',
                                  name='convolutional3D4')(convolutional3D3)
        return imageInputs, convolutional3D4

    def Convolutional3DSimple(self, classes, input_shape1):
        model = Sequential()
        # FIRST LAYER
        model.add(Conv3D(64, (3, 3, 3),
                         activation='relu',
                         padding='same',
                         input_shape=input_shape1))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(1, 2, 2),
                               strides=(1, 2, 2),
                               padding='valid',
                               name='pool1'))
        # SECOND LAYER
        model.add(Conv3D(128, (3, 3, 3),
                         activation='relu',
                         padding='same',
                         input_shape=input_shape1))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(1, 2, 2),
                               strides=(1, 2, 2),
                               padding='valid',
                               name='pool2'))
        # THIRD LAYER
        model.add(Conv3D(256, (3, 3, 3),
                         activation='relu',
                         padding='same',
                         input_shape=input_shape1))
        model.add(Conv3D(256, (3, 3, 3),
                         activation='relu',
                         padding='same',
                         input_shape=input_shape1))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(1, 2, 2),
                               strides=(1, 2, 2),
                               padding='valid',
                               name='pool3'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', name='fc6'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu', name='fc7'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        return model

    def StepDecay(self, epoch):
        initialLRate = 0.001
        drop = 0.1
        epochDrop = 10.0
        lRate = initialLRate * math.pow(drop, math.floor((1 + epoch) / epochDrop))
        return lRate

    def LrPolynomialDecay(self, globalStep):
        LearingRate = 0.003
        endLearningRate = 0.00001
        decaySteps = self.stepsPerEpoch * self.nbEpoch
        power = 0.9
        p = float(globalStep) / float(decaySteps)
        lr = (LearingRate - endLearningRate) * np.power(1 - p, power) + endLearningRate
        return lr

    def Train(self, model, train_generator, validation_generator, nb_epoch, steps_per_epoch, validation_steps,
              save_model_path, exp):
        LRate = LearningRateScheduler(self.LrPolynomialDecay, steps_per_epoch)
        optimizer = SGD(momentum=0.9, decay=0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        earlyStopper = EarlyStopping(patience=4)
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(save_model_path, exp, '{epoch:03d}-{ValidationLoss:.3f}.h5'),
            verbose=1, save_best_only=True)
        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=int(nb_epoch),
                                      verbose=1,
                                      callbacks=[earlyStopper, checkpointer, LRate],
                                      validation_data=validation_generator,
                                      validation_steps=validation_steps)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # make folder to save the diagrams
        if not os.path.isdir(os.path.join(save_model_path, exp + "_conf_matrix")):
            os.mkdir(os.path.join(save_model_path, exp + "_conf_matrix"))

        if not os.path.isdir(os.path.join(save_model_path, exp + "_diagrams")):
            os.mkdir(os.path.join(save_model_path, exp + "_diagrams"))
        nameloss = "model train vs validation loss"
        nameacc = "model train vs validation accuracy"
        # for loss
        fig_folder = os.path.join(save_model_path, exp + "_diagrams", exp + "_" + "loss")
        createplot(loss, val_loss, nameloss, fig_folder, "loss")
        # for accuracy
        fig_folder = os.path.join(save_model_path, exp + "_diagrams", exp + "_" + "accuracy")
        createplot(acc, val_acc, nameacc, fig_folder, "accuracy")
