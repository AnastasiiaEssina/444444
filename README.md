# 🧭 Меридиан · Intent Router

Streamlit-демо финальной модели классификации пользовательских намерений из командного проекта.

## Возможности

- классификация одной фразы;
- учёт истории диалога;
- загрузка CSV / JSON / JSONL;
- работа с датасетом с метками и без меток;
- Accuracy, Macro-F1, Precision, Recall;
- метрики по классам и матрица ошибок;
- выгрузка результатов в CSV;
- описание всех 8 intent-классов.

## Структура

```text
.
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   ├── intent_classifier_v2.joblib
│   ├── gigaevo_pca.joblib
│   └── needs_clarification_classifier_augmented.joblib
└── model_utils/
    ├── improve_context_model.py
    └── prepare_embeddings.py
```

Папка `model/` обязательна: в ней лежат сохранённые артефакты, необходимые для
работы приложения. Она находится рядом с `streamlit_app.py` и должна быть
добавлена в тот же репозиторий.

## Запуск локально

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## GigaChat API

Для классификации новых фраз нужен `GIGACHAT_API_KEY`.

Не добавляйте ключ в Git. Для Streamlit Cloud используйте Secrets:

```toml
GIGACHAT_API_KEY = "..."
```

## Формат собственного датасета

Обязательное поле:

- `text` — текущее сообщение пользователя.

Необязательные:

- `history` — список `{role, content}` или JSON-строка;
- `intent` — правильная метка, если нужны метрики.
- `needs_clarification` — необязательная булева метка для дополнительной проверки неоднозначности.

Допустимые метки:

`greeting`, `capabilities`, `gratitude`, `data_catalog`, `pivot_table`, `technical`, `support`, `no_rag`.

## Результаты модели

Финальная модель команды получила Macro-F1 0.9860 на validation dataset_3. Значение 1.0000 на hard-dev/regression-наборе не является независимой оценкой на новых данных.

## needs_clarification

В hybrid-режиме приложение возвращает дополнительный флаг
`needs_clarification`. Основной ансамбль сначала определяет один из восьми
intent, и только для `data_catalog` отдельный бинарный классификатор оценивает,
смешаны ли в запросе поиск объекта данных и просьба объяснить тему. Он
использует GigaChat EmbeddingsGigaR, PCA до 1 024 компонент и Logistic
Regression; в артефакте также сохранена TF-IDF-ветка для текстовых вариантов
ансамбля. Для любого другого intent флаг принудительно равен `false`.

Локальный режим не вычисляет этот флаг, потому что выбранная версия детектора
использует embedding-ветку. На regression-подмножестве `examples.jsonl` при
обязательном intent-gate получены precision 1.000, recall 0.500 и F1 0.667;
это условная оценка при корректном первичном intent-маршруте.
