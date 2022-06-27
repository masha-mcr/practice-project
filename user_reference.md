## Руководство по использованию

1. Склонировать репозиторий с помощью git clone

2. Установить зависимости с помощью poetry (предварительно должен быть установлен сам poetry)


    poetry install --no-dev 

3. Доступные команды

**Обучение модели, её тестирование на тестовом наборе**

    poetry run fit_predict

Параметры:

    Usage: fit_predict [OPTIONS]

    Options:

      -m, --model-id TEXT   //model name, must be unique
      -e, --epochs INTEGER          [default: 30]
      -b, --batch-size INTEGER      [default: 128]
      -r, --ratio <FLOAT FLOAT>...  [default: 0.2, 0.1]     //test : train+val ratio, val : train ratio
      --help                        Show this message and exit.

Пример команды:

    poetry run fit_predict -m 12345model -e 20 -r 0.3 0.15

**Обучение модели, её тестирование на загруженном изображении**

    poetry run fit_predict_single

Параметры:

    Usage: fit_predict_single [OPTIONS]

    Options:
      -m, --model-id TEXT          
      -e, --epochs INTEGER          [default: 30]
      -b, --batch-size INTEGER      [default: 128]
      -r, --ratio <FLOAT FLOAT>...  [default: 0.2, 0.1]
      -i, --image-path FILE     //path to image
      -l, --label INTEGER       //image label (0 - healthy, 1 - has covid)
      --help                        Show this message and exit.

**Загрузка обученной модели, её тестирование на тестовом наборе**

    poetry run load_predict

Параметры:
    
    Usage: load_predict [OPTIONS]
    
    Options:
      -m, --model-id TEXT       //id of model to load
                                //if not specified, loading the last saved model
      -r, --ratio FLOAT    [default: 0.2]       // how much data goes into testing set
      --help                        Show this message and exit.

**Загрузка обученной модели, её тестирование на загруженном изображении**

    poetry run load_predict_single

Параметры:
    
    Usage: load_predict_single [OPTIONS]
    
    Options:
      -m, --model-id TEXT       //id of model to load
                                //if not specified, loading the last saved model
      -i, --image-path FILE     //path to image
      -l, --label INTEGER       //image label (0 - healthy, 1 - has covid)
      --help                 Show this message and exit.

Посмотреть справку к каждой команде можно с помощью

    poetry run <command_name> --help

4. Посмотреть записи экспериментов в mlflow можно командой


    mlflow ui
(из корневой папки)

5. Обученные модели хранятся в папке ``models``, по названию модели можно также найти историю ее обучения (файл .pickle) и графики тренировочных метрик.