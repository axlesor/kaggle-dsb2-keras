from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from model import get_model
from utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation


def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('data/X_train.npy')
    y = np.load('data/y_train.npy')

    X = X.astype(np.float32)
    X /= 255

    seed = np.random.randint(1, 10e6)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X, y


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test


def train():
    """
    Training systole and diastole models.
    """
    print('Loading current Model and the latest weights...')
    model_systole = get_model()
    model_diastole = get_model()
#    model_systole = model_from_json(open('model_systole_architect.json').read())
    model_systole.load_weights('weights_systole_best.hdf5')

#    model_diastole = model_from_json(open('model_diastole_architect.json').read())
    model_diastole.load_weights('weights_diastole_best.hdf5')

    # load val losses to use as sigmas for CDF
    with open('val_loss.txt', mode='r') as f:
        val_loss_systole = float(f.readline())
        val_loss_diastole = float(f.readline())

    print('Loading training data...')
    X, y = load_train_data()

    print('Pre-processing images...')
    X = preprocess(X)

    # split to training and test
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.2)

    nb_iter = 15
    epochs_per_iter = 1
    batch_size = 32
    calc_crps = 1  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_systole = val_loss_systole
    min_val_loss_diastole = val_loss_diastole

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)

        print('Augmenting images - rotations')
        X_train_aug = rotation_augmentation(X_train, 15)
        print('Augmenting images - shifts')
        X_train_aug = shift_augmentation(X_train_aug, 0.1, 0.1)

        print('Fitting systole model...')
        hist_systole = model_systole.fit(X_train_aug, y_train[:, 0], shuffle=True, nb_epoch=epochs_per_iter,
                                         batch_size=batch_size, validation_data=(X_test, y_test[:, 0]))

        print('Fitting diastole model...')
        hist_diastole = model_diastole.fit(X_train_aug, y_train[:, 1], shuffle=True, nb_epoch=epochs_per_iter,
                                           batch_size=batch_size, validation_data=(X_test, y_test[:, 1]))

        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = hist_systole.history['loss'][-1]
        loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]
        val_loss_diastole = hist_diastole.history['val_loss'][-1]

        if calc_crps > 0 and i % calc_crps == 0:
            print('Evaluating CRPS...')
            pred_systole = model_systole.predict(X_train, batch_size=batch_size, verbose=1)
            pred_diastole = model_diastole.predict(X_train, batch_size=batch_size, verbose=1)
            val_pred_systole = model_systole.predict(X_test, batch_size=batch_size, verbose=1)
            val_pred_diastole = model_diastole.predict(X_test, batch_size=batch_size, verbose=1)

            # CDF for train and test data (actually a step function)
            cdf_train = real_to_cdf(np.concatenate((y_train[:, 0], y_train[:, 1])))
            cdf_test = real_to_cdf(np.concatenate((y_test[:, 0], y_test[:, 1])))

            # CDF for predicted data
            cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
            cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)
            cdf_val_pred_systole = real_to_cdf(val_pred_systole, val_loss_systole)
            cdf_val_pred_diastole = real_to_cdf(val_pred_diastole, val_loss_diastole)

            # evaluate CRPS on training data
            crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
            print('CRPS(train) = {0}'.format(crps_train))

            # evaluate CRPS on test data
            crps_test = crps(cdf_test, np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
            print('CRPS(test) = {0}'.format(crps_test))

        print('Saving weights...')
        # save weights so they can be loaded later
        model_systole.save_weights('weights_systole.hdf5', overwrite=True)
        model_diastole.save_weights('weights_diastole.hdf5', overwrite=True)

        # for best (lowest) val losses, save weights
        if val_loss_systole < min_val_loss_systole:
            min_val_loss_systole = val_loss_systole
            model_systole.save_weights('weights_systole_best.hdf5', overwrite=True)

        if val_loss_diastole < min_val_loss_diastole:
            min_val_loss_diastole = val_loss_diastole
            model_diastole.save_weights('weights_diastole_best.hdf5', overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('val_loss.txt', mode='w+') as f:
            f.write(str(min_val_loss_systole))
            f.write('\n')
            f.write(str(min_val_loss_diastole))


train()
