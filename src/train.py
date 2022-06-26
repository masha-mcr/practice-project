from data_loader import load_data, train_val_test_split
from test_evaluate import evaluate
import pickle

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt



def fit_predict():
    images, labels = load_data(data_dir='../data/COVID-19_Radiography_Dataset')
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(images, labels, ratio=(0.2, 0.1))
    model_id = 'primary_model'
    model = build_model()
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    batch_size = 64
    epochs = 20
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    history = model.fit(x_train, y_train,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val),
                            callbacks = [es])
    model.save(f'../models/{model_id}')

    with open(f'{model_id}_history.pickle', 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)


    test_metric = evaluate(model, x_test, y_test, save_metric=True)
    plot_training_metrics(history)


#    with open('filename.pickle', 'rb') as handle:
#        b = pickle.load(handle)


def build_model():
    cnn_model = Sequential()
    cnn_model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='linear', input_shape=(70, 70, 3), padding='same'))
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(MaxPooling2D((2, 2), padding='same'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='linear', padding='same'))
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(MaxPooling2D((2, 2), padding='same'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='linear', padding='same'))
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(MaxPooling2D((2, 2), padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=16, activation='linear'))
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(Dropout(0.1))
    cnn_model.add(Dense(units=2, activation='softmax'))

    return cnn_model


def plot_training_metrics(history, test_metrics=None):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    if test_metrics:
        test_accuracy = test_metrics[1]
        test_accuracy = test_metrics[1]
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, label='Training accuracy')
    plt.plot(epochs, val_accuracy, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

fit_predict()