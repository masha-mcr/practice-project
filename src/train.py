from data_loader import load_data, train_val_test_split

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping

#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)


def train():
    images, labels = load_data(data_dir='../data/COVID-19_Radiography_Dataset')
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(images, labels, ratio=(0.2, 0.1))
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)
    model = build_model()
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    batch_size = 64
    epochs = 20
    num_classes = 2
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    history = model.fit(x_train, y_train,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val),
                            callbacks = [es])


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

train()