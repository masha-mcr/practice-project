import click
import os

import numpy as np
from .test_evaluate import evaluate, classification_report, predict
from .data_loader import load_data, train_val_test_split, read_image
import keras.models


@click.command()
@click.option(
    "-m",
    "--model-id",
    default=None,
    type=str,
    show_default=True,
)
def load_predict(model_id):
    click.secho('Loading model...', fg='green')
    model = get_model(model_id)
    model.summary()

    images, labels = load_data(data_dir='data/COVID-19_Radiography_Dataset')
    x_test, y_test = train_val_test_split(images, labels, ratio=(0.3, None), test_only=True)

    evaluate(model, x_test, y_test, save_metric=False)
    y_pred = predict(model, x_test)
    click.echo(classification_report(model, x_test, y_test, y_pred))


@click.command()
@click.option(
    "-m",
    "--model-id",
    default=None,
    type=str,
    show_default=True,
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
def load_predict_single(model_id, image_path, label):
    click.echo(image_path)
    image_read = read_image(image_path)
    model = get_model(model_id)
    label_pred = np.argmax(predict(model, x_test=[image_read]), axis=1)
    if label_pred == label:
        click.secho(f'Predicted: {label_pred}', fg='green')
    else:
        click.secho(f'Predicted: {label_pred}', fg='red')
    click.secho(f'Expected: {label}', fg='green')


def get_last_model():
    saved_models = [os.path.join('models', f) for f in os.listdir('models') if os.path.isdir(os.path.join('models', f))]
    print(saved_models)
    print(os.path.getmtime('models/10ep_128b'))
    saved_models = sorted(saved_models, key=os.path.getmtime)
    last_model_id = saved_models[-1].replace('models\\', '')
    return last_model_id


def get_model(model_id):
    if not model_id:
        model_id = get_last_model()
    model = keras.models.load_model(f'models/{model_id}')
    return model
