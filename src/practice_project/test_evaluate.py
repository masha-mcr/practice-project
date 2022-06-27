import numpy as np
import sklearn.metrics
import click


def evaluate(model, x_test, y_test, save_metric=False):
    click.secho("Evaluation...", fg="green")
    test_eval = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", test_eval[0])
    print("Test accuracy:", test_eval[1])
    if save_metric:
        return test_eval


def predict(model, x_test):
    predicted_classes = model.predict(x_test)
    return np.round(predicted_classes)


def classification_report(model, x_test, y_test, y_pred=None, return_dict=False):
    target_names = ["Class {}".format(i) for i in range(2)]
    if y_pred.all():
        predicted_classes = y_pred
    else:
        predicted_classes = predict(model, x_test)
    return sklearn.metrics.classification_report(
        y_test, predicted_classes, target_names=target_names, output_dict=return_dict
    )
