import numpy as np


def evaluate(model, x_test, y_test, save_metric=False):
    test_eval = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    if save_metric:
        return test_eval


def predict(model, x_test):
    predicted_classes_OHE = model.predict(x_test)
    predicted_classes = np.argmax(np.round(predicted_classes_OHE), axis=1)
