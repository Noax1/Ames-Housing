# Система оценки стоимости недвижимости (Ames Housing)

* **Команда:** Квартирный ответ (БИВТ-23-АЭ-1)
* **Задача:** Регрессия - предсказание цены продажи дома (`SalePrice`, USD)
* **Модель:** `Pipeline` (`ColumnTransformer` + `TransformedTargetRegressor(LinearRegression, log1p/expm1)`)
* **Разделение данных:** обучение `YrSold <= 2009` (1285 домов), валидация `YrSold == 2010` (175 домов)
* **Метрики на тесте:** `MAE = 15 310.92 $`, `RMSE = 22 641.32 $`, `R² = 0.9203`

## 1. Установка зависимостей и обучение модели
```bash
pip install -r requirements.txt
python train.py
```
Команда создаёт в папке `artifacts/` файлы `model.joblib`, `meta.json` и `etalon.json`.

## 2. Запуск автотестов
```bash
python -m pytest
```

## 3. Локальный запуск сервиса
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Интерактивная документация Swagger UI доступна по адресу: http://localhost:8000/docs

## 4. Сборка и запуск Docker-контейнера
```bash
docker build -t ames-housing:v1.0.0 .
docker run -d -p 8000:8000 --name housing-api ames-housing:v1.0.0
```

Проверка эндпоинтов:
* `GET http://localhost:8000/health`
* `GET http://localhost:8000/model-info`
* `POST http://localhost:8000/predict`

## 5. Откат на предыдущую версию (Rollback)
```bash
docker stop housing-api
docker rm housing-api
docker run -d -p 8000:8000 --name housing-api ames-housing:v1.0.0
```