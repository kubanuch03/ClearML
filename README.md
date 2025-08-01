## 🚀 Развертывание кастомной модели с ClearML Serving: Полное руководство

Добро пожаловать в руководство по использованию clearml-serving для развертывания ваших собственных моделей! ClearML — это мощная MLOps-платформа с открытым исходным кодом, которая позволяет автоматизировать, отслеживать и оркестрировать весь жизненный цикл машинного обучения.  Это руководство проведет вас через все шаги: от первоначальной настройки проекта до отправки тестовых запросов на вашу развернутую модель. <br>

ClearML Serving — это утилита командной строки для развертывания моделей и их оркестрации. Она позволяет легко разворачивать модели машинного и глубокого обучения, включая код предварительной обработки, в кластере Kubernetes или в кастомных контейнерных решениях. 

## 🚀 Инициализация проекта с `clearml-init`
Для подключения проекта к [ClearML](https://clear.ml/) выполните инициализацию:

```bash
clearml-init
```

```bash

$ clearml-init

ClearML SDK setup process

Please create new credentials through your ClearML Server web UI:
    Settings -> Workspace -> Create new credentials

In your browser:
    Web: https://app.clear.ml

New API Access Key: AKxxxxxxxxxxxxxxxxxx
New API Secret Key: SKxxxxxxxxxxxxxxxxxx
ClearML API server [https://api.clear.ml]:
ClearML Web server [https://app.clear.ml]:
ClearML Files server [https://files.clear.ml]:

Configuration stored in /home/youruser/.clearml.conf
ClearML setup completed successfully.
```



## 🛰️ Создание сервиса с `clearml-serving create`
Команда `clearml-serving create` используется для создания нового **инференс-сервиса** в ClearML Serving.
### 📌 Синтаксис
```bash
clearml-serving create --name "Detect Serving Service"
```


## 🧠 Добавление кастомной модели в ClearML Serving

После создания сервиса можно привязать модель для инференса с помощью команды:
```bash
clearml-serving --id "<ID Сервиса который получили с команды сверху>" model add \
--engine custom \
--endpoint "car-detector" \
--project "Car Detection Project" \
--name "yolov8s_cars_finetuned3" \
--preprocess "<Название файла для preprocess>.py"
```


## 🔍 Что делает эта команда:

| 🧩 Аргумент        | 📝 Описание                                                                  |
|------------------|--------------------------------------------------------------------------|
| `--id`           | ID сервиса, полученный при создании (`clearml-serving create`)          |
| `model add`      | Указывает, что нужно добавить модель к сервису                          |
| `--engine custom`| Тип движка для инференса. Значение custom указывает, что для обработки запросов будет использоваться ваш собственный Python-код. Это обеспечивает максимальную гибкость: вы можете определить собственную логику загрузки модели, предобработки (preprocess) и постобработки (postprocess) данных. Альтернативами могут быть оптимизированные движки, такие как triton.Python-код                             |
| `--endpoint`     | Имя конечной точки (endpoint). Это публичный URL-путь, по которому модель будет доступна для запросов. В данном примере конечный URL будет выглядеть как http://localhost:8080/serve/car-detector. Каждый эндпоинт в рамках одного сервиса должен быть уникальным.|
| `--project`      | Название проекта в ClearML, где зарегистрирована модель                 |
| `--name`         |Имя модели. Это точное имя модели, которое отображается в разделе "Models" вашего проекта в веб-интерфейсе ClearML. Имя должно полностью совпадать, иначе сервис не сможет найти нужный артефакт.  |
| `--preprocess`   |Файл с пользовательской логикой. Путь к Python-файлу, который содержит класс для обработки модели. Для движка custom этот файл обязателен. ClearML Serving ожидает, что в этом файле будет класс (по умолчанию `Preprocess`), содержащий как минимум два метода: `load()` для загрузки модели в память и `predict()` для выполнения инференса на входных данных.


## 🐳 Запуск ClearML Serving через Docker Compose

После настройки модели и сервиса, запустите инференс-окружение с помощью Docker Compose:
```bash
cd clearml-serving/docker
docker compose up --build -d
```

## 📌 Чтобы проверить логи:
```bash
docker compose logs -f clearml-serving-inference
```


## 🧪 Отправка тестового запроса на распознавание

После запуска сервиса можно протестировать работу модели, отправив изображение на инференс с помощью скрипта:

```bash
cd test_img/
python3 send_request_recognize.py
```

## 🌐 Доступ к API
Ваша модель теперь доступна через REST API. Вы можете отправлять POST-запросы на следующий URL для получения предсказаний:<br>
http://localhost:8080/serve/car-detector <br>
Стандартный порт веб-сервера ClearML — 8080.<br> 
Вы также можете получить доступ к веб-интерфейсу ClearML по адресу http://localhost:8080/docs, чтобы отслеживать состояние ваших сервисов и эндпоинтов.

<!-- # ```bash
# docker stop $(docker ps -a -q)
# docker rm $(docker ps -a -q)
# ```   -->

 






 