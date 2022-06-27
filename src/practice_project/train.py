from .data_loader import load_data, train_val_test_split, read_image
from .test_evaluate import evaluate, classification_report, predict
import pickle

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np

import mlflow
import click

@click.command()
@click.option(
    "-m",
    "--model-id",
    type=str,
)
@click.option(
    "-e",
    "--epochs",
    default=30,
    type=int,
    show_default=True,
)
@click.option(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    show_default=True,
)
@click.option(
    "-r",
    "--ratio",
    default=(0.2, 0.1),
    type=(float, float),
    show_default=True
)
def fit_predict(model_id, epochs, batch_size, ratio):

    images, labels = load_data(data_dir="data/COVID-19_Radiography_Dataset")
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        images, labels, ratio=ratio)

    model = build_model()
    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    with mlflow.start_run(run_name=model_id):
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=[es],
        )

        save_model = save_dialogue(model, history, model_id)

        plot_training_metrics(history, model_id, save_model)

        test_metric = evaluate(model, x_test, y_test, save_metric=True)
        report = classification_report(model, x_test, y_test, return_dict=True)
        click.echo(classification_report(model, x_test, y_test))

        click.secho("Logging...", fg="green")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch", batch_size)
        mlflow.log_metric("train_acc", history.history["accuracy"][-1])
        mlflow.log_metric("train_loss", history.history["loss"][-1])
        mlflow.log_metric("test_acc", test_metric[1])
        mlflow.log_metric("test_loss", test_metric[0])
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1", report["weighted avg"]["f1-score"])
        if save_model:
            mlflow.log_artifact(f"models/{model_id}_acc_curve.png")
            mlflow.log_artifact(f"models/{model_id}_loss_curve.png")

@click.command()
@click.option(
    "-m",
    "--model-id",
    type=str,
)
@click.option(
    "-e",
    "--epochs",
    default=30,
    type=int,
    show_default=True,
)
@click.option(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    show_default=True,
)
@click.option(
    "-r",
    "--ratio",
    default=(0.2, 0.1),
    type=(float, float),
    show_default=True
)
@click.option(
    "-i",
    "--image-path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-l",
    "--label",
    type=int,
)
def fit_predict_single(model_id, epochs, batch_size, ratio, image_path=None, label=None):

    images, labels = load_data(data_dir="data/COVID-19_Radiography_Dataset")
    x_train, x_val, _, y_train, y_val, _ = train_val_test_split(
        images, labels, ratio=ratio)

    x_test = read_image(image_path).reshape((-1, 70, 70, 3))

    model = build_model()
    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    with mlflow.start_run(run_name=model_id):
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=[es],
        )

        save_model = save_dialogue(model, history, model_id)

        plot_training_metrics(history, model_id, save_model)

        y_pred = np.argmax(predict(model, x_test=[x_test]), axis=1)
        if y_pred == label:
            click.secho(f"Predicted: {y_pred}", fg="green")
        else:
            click.secho(f"Predicted: {y_pred}", fg="red")
        click.secho(f"Expected: {label}", fg="green")

        click.secho("Logging...", fg="green")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch", batch_size)
        mlflow.log_metric("train_acc", history.history["accuracy"][-1])
        mlflow.log_metric("train_loss", history.history["loss"][-1])
        if save_model:
            mlflow.log_artifact(f"models/{model_id}_acc_curve.png")
            mlflow.log_artifact(f"models/{model_id}_loss_curve.png")


def build_model():
    cnn_model = Sequential()
    cnn_model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="linear",
            input_shape=(70, 70, 3),
            padding="same",
        )
    )
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(MaxPooling2D((2, 2), padding="same"))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(
        Conv2D(filters=64, kernel_size=(3, 3), activation="linear", padding="same")
    )
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(MaxPooling2D((2, 2), padding="same"))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(
        Conv2D(filters=32, kernel_size=(3, 3), activation="linear", padding="same")
    )
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(MaxPooling2D((2, 2), padding="same"))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=16, activation="linear"))
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(Dropout(0.1))
    cnn_model.add(Dense(units=2, activation="softmax"))

    return cnn_model


def plot_training_metrics(history, model_id, save_model):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, label="Training accuracy")
    plt.plot(epochs, val_accuracy, label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    if save_model:
        plt.savefig(f"models/{model_id}_acc_curve.png")

    plt.figure()
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    if save_model:
        plt.savefig(f"models/{model_id}_loss_curve.png")


def save_dialogue(model, history, model_id):
    click.echo('Save model? [y/n] ', nl=False)
    c = click.getchar()
    click.echo()
    if c == 'y':
        save_model = True
    elif c == 'n':
        save_model = False
    else:
        click.echo('Invalid input, model will not be saved')
        save_model = False

    if save_model:
        model.save(f"models/{model_id}")
        with open(f"models/{model_id}_history.pickle", "wb") as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return save_model
