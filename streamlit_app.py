from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

# Streamlit Cloud exposes root-level secrets through st.secrets.
# Mirror the key into the environment for the optional GigaChat path.
try:
    if "GIGACHAT_API_KEY" in st.secrets and not os.getenv("GIGACHAT_API_KEY"):
        os.environ["GIGACHAT_API_KEY"] = st.secrets["GIGACHAT_API_KEY"]
except Exception:
    pass

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
UTILS_DIR = ROOT / "model_utils"
sys.path.insert(0, str(UTILS_DIR))

from improve_context_model import LABELS, apply_policy, softmax, transform_text  # noqa: E402
from prepare_embeddings import GigaChatEmbeddings, format_input  # noqa: E402

st.set_page_config(
    page_title="Меридиан · Intent Router",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

LABEL_INFO = {
    "greeting": ("Приветствие", "Начало диалога, приветствие или короткое обращение."),
    "capabilities": ("Возможности", "Вопрос о том, что умеет система и какие задачи она поддерживает."),
    "gratitude": ("Благодарность", "Спасибо, благодарность или положительная реакция на ответ."),
    "data_catalog": ("Каталог данных", "Вопрос о доступных данных, наборах, полях и источниках."),
    "pivot_table": ("Сводная таблица", "Запрос на построение, изменение или настройку сводной таблицы."),
    "technical": ("Технический вопрос", "Проблема с работой системы, ошибкой, форматом или выполнением действия."),
    "support": ("Поддержка", "Явный запрос соединить с оператором или живым специалистом."),
    "no_rag": ("Без внешнего поиска", "Запрос можно выполнить по уже имеющейся информации диалога, без внешнего поиска."),
}

@st.cache_resource(show_spinner=False)
def load_artifacts():
    paths = {
        "intent_classifier_v2.joblib": MODEL_DIR / "intent_classifier_v2.joblib",
        "gigaevo_pca.joblib": MODEL_DIR / "gigaevo_pca.joblib",
        "needs_clarification_classifier_augmented.joblib": (
            MODEL_DIR / "needs_clarification_classifier_augmented.joblib"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Не найдены файлы модели в "
            f"{MODEL_DIR}: {', '.join(missing)}. "
            "В репозитории рядом с streamlit_app.py должна быть папка model/."
        )
    artifact = joblib.load(paths["intent_classifier_v2.joblib"])
    pca = joblib.load(paths["gigaevo_pca.joblib"])
    clarifier = joblib.load(paths["needs_clarification_classifier_augmented.joblib"])
    return artifact, pca, clarifier

@st.cache_resource(show_spinner=False)
def get_embedder():
    return GigaChatEmbeddings()


def clarification_probabilities(
    artifact: dict[str, Any], embedding: np.ndarray, rows: list[dict[str, Any]]
) -> np.ndarray:
    """Score catalog/technical ambiguity using the embedding already requested."""
    dense_probability = artifact["dense_model"].predict_proba(embedding)[:, 1]
    dense_weight = float(artifact["dense_weight"])
    if dense_weight == 1.0:
        return dense_probability
    formatted = [format_input(row) for row in rows]
    word = artifact["sparse_word_vectorizer"]
    char = artifact["sparse_char_vectorizer"]
    text_features = sparse.hstack(
        [word.transform(formatted), char.transform(formatted)], format="csr"
    )
    sparse_probability = artifact["sparse_model"].predict_proba(text_features)[:, 1]
    return dense_weight * dense_probability + (1.0 - dense_weight) * sparse_probability


def _build_result_from_scores(rows: list[dict[str, Any]], score: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    # Convert final routing scores to a user-facing relative distribution.
    display = np.exp(score - score.max(axis=1, keepdims=True))
    display /= display.sum(axis=1, keepdims=True)

    order = np.argsort(score, axis=1)[:, ::-1]
    results: list[dict[str, Any]] = []
    for i, indices in enumerate(order):
        top = int(indices[0])
        second = int(indices[1])
        results.append(
            {
                "intent": LABELS[top],
                "score": float(display[i, top]),
                "top2": [
                    {"intent": LABELS[top], "score": float(display[i, top])},
                    {"intent": LABELS[second], "score": float(display[i, second])},
                ],
            }
        )
    return results, display


def classify_rows(
    rows: list[dict[str, Any]],
    use_embeddings: bool = False,
) -> tuple[list[dict[str, Any]], np.ndarray, str]:
    artifact, pca, clarifier = load_artifacts()

    # The context branch is fully local: it uses the saved TF-IDF vectorizers,
    # LinearSVC and routing policy from the team's v2 artifact.
    text_features = transform_text(rows, artifact["vectorizers"])
    temperature = float(artifact["config"]["temperature"])
    context_prob = softmax(
        artifact["context_model"].decision_function(text_features), temperature
    )

    if not use_embeddings:
        score = np.log(np.clip(context_prob, 1e-9, 1.0))
        if artifact["config"].get("policy", False):
            score = apply_policy(rows, score)
        results, display = _build_result_from_scores(rows, score)
        for result in results:
            # The selected clarification model needs GigaChat embeddings. A
            # missing score is more honest than a false negative in local mode.
            result["needs_clarification"] = None
        return results, display, "local"

    # Optional hybrid mode: the original final v2 path with GigaChat EmbeddingsGigaR.
    embedder = get_embedder()
    formatted = [format_input(row) for row in rows]
    raw = embedder.embed(formatted)
    embedding = np.asarray(pca.transform(raw), dtype=np.float32)
    embedding_prob = artifact["base_model"].predict_proba(embedding)

    alpha = float(artifact["config"]["embedding_weight"])
    probability = np.clip(
        alpha * embedding_prob + (1.0 - alpha) * context_prob,
        1e-9,
        1.0,
    )
    score = np.log(probability)
    if artifact["config"].get("policy", False):
        score = apply_policy(rows, score)

    results, display = _build_result_from_scores(rows, score)
    ambiguity_probability = clarification_probabilities(clarifier, embedding, rows)
    threshold = float(clarifier["threshold"])
    for index, result in enumerate(results):
        # CASE.md permits this flag only for the data_catalog route. Gating by
        # the primary model output prevents impossible intent/flag pairs.
        if result["intent"] == "data_catalog":
            result["needs_clarification"] = bool(ambiguity_probability[index] >= threshold)
            result["needs_clarification_score"] = float(ambiguity_probability[index])
        else:
            result["needs_clarification"] = False
            result["needs_clarification_score"] = None
    return results, display, "hybrid"


def parse_history_text(value: str) -> list[dict[str, str]]:
    history = []
    for line in value.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            history.append({"role": "user", "content": line})
            continue
        role, content = line.split(":", 1)
        role = role.strip().lower()
        if role in {"assistant", "ассистент", "a"}:
            role = "assistant"
        else:
            role = "user"
        history.append({"role": role, "content": content.strip()})
    return history


def parse_uploaded(file) -> pd.DataFrame:
    raw = file.getvalue()
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if name.endswith((".jsonl", ".ndjson")):
        rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    if name.endswith(".json"):
        data = json.loads(raw.decode("utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("data", data.get("rows", [data]))
        return pd.DataFrame(data)
    raise ValueError("Поддерживаются только CSV, JSON и JSONL.")


def normalize_history(value: Any) -> list[dict[str, str]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else parse_history_text(value)
        except json.JSONDecodeError:
            return parse_history_text(value)
    return []


def normalize_optional_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "да"}:
            return True
        if normalized in {"false", "0", "no", "нет"}:
            return False
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    return None


def dataframe_to_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "text" not in df.columns:
        raise ValueError("В датасете обязательно должен быть столбец `text`.")
    rows = []
    for _, record in df.iterrows():
        row = {"text": str(record["text"]), "history": normalize_history(record.get("history", []))}
        if "intent" in df.columns and pd.notna(record.get("intent")):
            row["intent"] = str(record["intent"])
        rows.append(row)
    return rows


def render_prediction(result: dict[str, Any]):
    label = result["intent"]
    title, description = LABEL_INFO[label]
    st.markdown(f"<div class='result-label'>{title}</div>", unsafe_allow_html=True)
    st.caption(f"`{label}` · {description}")

    c1, c2 = st.columns(2)
    c1.metric("Основной результат", title)
    c2.metric("Относительный score", f"{result['score']:.1%}")

    clarification = result.get("needs_clarification")
    if clarification is None:
        st.caption("Проверка needs_clarification доступна в режиме GigaChat Embeddings.")
    elif clarification:
        st.warning("Нужно уточнение: запрос может означать поиск объекта данных или объяснение темы.")
    else:
        st.success("Дополнительное уточнение по границе «каталог / объяснение» не требуется.")

    st.markdown("**Два наиболее вероятных класса**")
    for item in result["top2"]:
        name, _ = LABEL_INFO[item["intent"]]
        st.write(f"**{name}** (`{item['intent']}`) — {item['score']:.1%}")
        st.progress(min(max(item["score"], 0.0), 1.0))


def render_metrics(y_true, y_pred):
    labels_present = [label for label in LABELS if label in set(y_true) or label in set(y_pred)]
    st.markdown("### Основные метрики")
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
    b.metric("Macro-F1", f"{f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.3f}")
    c.metric("Macro Precision", f"{precision_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.3f}")
    d.metric("Macro Recall", f"{recall_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.3f}")

    report = classification_report(y_true, y_pred, labels=LABELS, target_names=[LABEL_INFO[x][0] for x in LABELS], output_dict=True, zero_division=0)
    metric_df = pd.DataFrame(report).T.loc[[LABEL_INFO[x][0] for x in labels_present if x in report]]
    st.markdown("### Метрики по классам")
    st.dataframe(metric_df[["precision", "recall", "f1-score", "support"]].round(3), use_container_width=True)

    st.markdown("### Матрица ошибок")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[LABEL_INFO[x][0] for x in LABELS], columns=[LABEL_INFO[x][0] for x in LABELS])
    st.dataframe(cm_df, use_container_width=True)

    errors = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            errors.append(i)
    if errors:
        st.markdown(f"### Ошибки классификации · {len(errors)}")
        st.caption("Первые 100 строк с несовпадением правильной и предсказанной метки.")
        error_df = pd.DataFrame({"№": [i + 1 for i in errors[:100]], "Правильная метка": [LABEL_INFO.get(y_true[i], (y_true[i],))[0] for i in errors[:100]], "Предсказание": [LABEL_INFO.get(y_pred[i], (y_pred[i],))[0] for i in errors[:100]]})
        st.dataframe(error_df, use_container_width=True, hide_index=True)
    else:
        st.success("Ошибок на загруженном датасете не обнаружено.")


def main():
    st.markdown("# 🧭 Меридиан · Intent Router")
    st.markdown("### Классификация пользовательских намерений с учётом контекста диалога")
    st.caption("Финальная модель команды · GigaChat EmbeddingsGigaR + контекстный LinearSVC")

    with st.sidebar:
        st.markdown("## Навигация")
        page = st.radio("Раздел", ["Демо", "Мой датасет", "Метрики модели", "8 классов", "О проекте"], label_visibility="collapsed")
        st.divider()
        st.markdown("**Режим классификации**")
        use_embeddings = st.checkbox(
            "Использовать гибридную модель GigaChat",
            value=True,
            help="Объединяет семантическую ветку GigaChat EmbeddingsGigaR и контекстную ветку LinearSVC.",
        )
        if use_embeddings:
            st.info("Активна гибридная конфигурация: GigaChat EmbeddingsGigaR + LinearSVC.")
        else:
            st.success("Активна локальная контекстная ветка.")
        st.caption("Локальный режим использует сохранённую контекстную ветку v2: TF-IDF + LinearSVC + policy.")
        st.caption("Флаг needs_clarification вычисляется только в hybrid-режиме: он использует отдельную embedding-модель.")

    if page == "Демо":
        st.markdown("## Попробуйте классификатор")
        st.info("Можно проверить отдельную фразу или показать, как история диалога меняет интерпретацию короткого запроса.")

        examples = {
            "Выберите пример": ("", ""),
            "Простая фраза": ("Привет!", ""),
            "Вопрос о возможностях": ("Какие данные ты умеешь анализировать?", ""),
            "Техническая проблема": ("Почему у меня не открывается таблица?", ""),
            "Контекстный пример": (
                "А теперь построй её по месяцам",
                "Ассистент: Я могу работать с данными и строить сводные таблицы.\nПользователь: Мне нужна таблица по продажам за год.",
            ),
        }
        selected = st.selectbox("Быстрый пример", list(examples))
        default_text, default_history = examples[selected]

        col1, col2 = st.columns([1.1, 1])
        with col1:
            text = st.text_area("Текущее сообщение пользователя", value=default_text, height=130, placeholder="Например: Как посмотреть доступные наборы данных?")
        with col2:
            history_text = st.text_area("История диалога · необязательно", value=default_history, height=130, placeholder="Ассистент: ...\nПользователь: ...")
            st.caption("Формат: одна реплика на строку — `Ассистент: ...` или `Пользователь: ...`")

        if st.button("Определить намерение", type="primary", use_container_width=True, disabled=not text.strip()):
            row = {"text": text.strip(), "history": parse_history_text(history_text)}
            with st.spinner("Анализируем сообщение…"):
                try:
                    result, _, mode = classify_rows([row], use_embeddings=use_embeddings)
                    st.divider()
                    if mode == "local":
                        st.caption("Режим: локальная контекстная ветка v2 · внешний API не используется")
                    else:
                        st.caption("Режим: исходная гибридная модель v2 · GigaChat EmbeddingsGigaR")
                    render_prediction(result[0])
                    with st.expander("Технический JSON-ответ"):
                        st.json({
                            "intent": result[0]["intent"],
                            "needs_clarification": result[0].get("needs_clarification"),
                            "top2": result[0]["top2"],
                        })
                except Exception as exc:
                    st.error("Не удалось выполнить классификацию.")
                    st.code(str(exc))
                    st.info("Для гибридного режима проверьте настройку GIGACHAT_API_KEY. Локальный режим доступен без API.")

    elif page == "Мой датасет":
        st.markdown("## Проверка собственного датасета")
        st.write("Загрузите CSV или JSONL. Если есть столбец `intent`, приложение рассчитает метрики. Если метки нет — сформирует только предсказания.")
        uploaded = st.file_uploader("Перетащите файл сюда", type=["csv", "json", "jsonl", "ndjson"])
        with st.expander("Формат файла"):
            st.code('{"text":"Какие данные доступны?","history":[],"intent":"data_catalog"}\n{"text":"Спасибо!","history":[]}', language="json")
            st.caption("Обязательный столбец: text. Необязательные: history и intent. history — JSON-массив сообщений или пустое значение.")

        if uploaded:
            try:
                df = parse_uploaded(uploaded)
                rows = dataframe_to_rows(df)
                st.success(f"Загружено строк: {len(rows):,}")
                st.dataframe(df.head(10), use_container_width=True)

                if st.button("Запустить классификацию датасета", type="primary", use_container_width=True):
                    progress = st.progress(0, text="Подготовка…")
                    all_results = []
                    batch_size = 32
                    mode = "hybrid" if use_embeddings else "local"
                    try:
                        for start in range(0, len(rows), batch_size):
                            batch = rows[start:start + batch_size]
                            results, _, _ = classify_rows(batch, use_embeddings=use_embeddings)
                            all_results.extend(results)
                            progress.progress(min((start + len(batch)) / len(rows), 1.0), text=f"Обработано {min(start + len(batch), len(rows)):,} из {len(rows):,}")
                        progress.empty()

                        output = df.copy()
                        output["predicted_intent"] = [r["intent"] for r in all_results]
                        output["predicted_score"] = [r["score"] for r in all_results]
                        output["predicted_needs_clarification"] = [
                            r.get("needs_clarification") for r in all_results
                        ]
                        st.session_state["dataset_output"] = output
                        st.session_state["dataset_results"] = all_results
                        st.session_state["dataset_mode"] = mode
                        st.success("Классификация завершена.")
                    except Exception as exc:
                        progress.empty()
                        st.error("Ошибка при классификации датасета.")
                        st.code(str(exc))

                if "dataset_output" in st.session_state:
                    output = st.session_state["dataset_output"]
                    results = st.session_state["dataset_results"]
                    st.markdown("### Результаты")
                    st.dataframe(output, use_container_width=True)
                    if st.session_state.get("dataset_mode") == "local":
                        st.caption("В локальном режиме `predicted_needs_clarification` не рассчитывается: для него нужна GigaChat embedding-ветка.")
                    st.download_button("⬇️ Скачать результаты CSV", output.to_csv(index=False).encode("utf-8-sig"), "intent_predictions.csv", "text/csv", use_container_width=True)
                    if "intent" in df.columns:
                        y_true = [str(x) for x in df["intent"].tolist()]
                        unknown = sorted(set(y_true) - set(LABELS))
                        if unknown:
                            st.error(f"Неизвестные метки в столбце intent: {unknown}. Допустимы только 8 меток модели.")
                        else:
                            y_pred = [r["intent"] for r in results]
                            render_metrics(y_true, y_pred)
                    if "needs_clarification" in df.columns:
                        pairs = [
                            (normalize_optional_bool(actual), predicted, result)
                            for actual, predicted, result in zip(
                                df["needs_clarification"].tolist(),
                                [r.get("needs_clarification") for r in results],
                                results,
                            )
                            if normalize_optional_bool(actual) is not None and predicted is not None
                        ]
                        if pairs:
                            y_flag = [item[0] for item in pairs]
                            p_flag = [item[1] for item in pairs]
                            st.markdown("### Метрики needs_clarification")
                            a, b, c = st.columns(3)
                            a.metric("Precision", f"{precision_score(y_flag, p_flag, zero_division=0):.3f}")
                            b.metric("Recall", f"{recall_score(y_flag, p_flag, zero_division=0):.3f}")
                            c.metric("F1", f"{f1_score(y_flag, p_flag, zero_division=0):.3f}")
                            impossible = sum(
                                bool(predicted) and result["intent"] != "data_catalog"
                                for _, predicted, result in pairs
                            )
                            st.caption(f"Невозможных сочетаний flag=true и intent≠data_catalog: {impossible}.")
                        elif st.session_state.get("dataset_mode") == "local":
                            st.caption("Метрики needs_clarification требуют hybrid-режим с GigaChat Embeddings.")
            except Exception as exc:
                st.error("Не удалось прочитать файл.")
                st.code(str(exc))

    elif page == "Метрики модели":
        st.markdown("## Результаты финальной модели")
        st.write("Метрики ниже взяты из финального эксперимента команды на dataset_3.")
        a, b, c = st.columns(3)
        a.metric("Validation Macro-F1", "0.9860")
        b.metric("Hard-dev Macro-F1", "1.0000")
        c.metric("Clarifier F1 · regression", "0.667")
        st.info("Hard-dev и regression-набор использовались в процессе разработки. Значение 1.0000 не следует трактовать как независимую оценку на новых данных.")
        st.markdown("### Что именно классифицирует модель")
        st.write("На вход подаются только история диалога и текущее сообщение пользователя. Служебные поля, метки и скрытые состояния в классификацию не передаются.")
        st.markdown("### Состав финальной модели")
        st.markdown("`EmbeddingsGigaR` → `PCA (1024)` → embedding-классификатор + `TF-IDF` контекста → `LinearSVC` → объединение результатов → policy")
        st.markdown("### Дополнительный флаг needs_clarification")
        st.write("После маршрутизации в `data_catalog` отдельная модель на GigaChat embeddings оценивает, не смешаны ли в запросе поиск объекта данных и просьба объяснить тему. На регулярной validation она получила F1 0.980, на независимом hard-dev — 0.975. В regression-проверке `examples.jsonl` при обязательном intent-gate: precision 1.000, recall 0.500, F1 0.667.")
        st.markdown("### Локальная контекстная ветка")
        st.write("При необходимости доступна сохранённая локальная ветка v2: TF-IDF текущей реплики и истории → LinearSVC → policy. Гибридная конфигурация дополняет её семантическими признаками GigaChat.")

    elif page == "8 классов":
        st.markdown("## 8 классов намерений")
        for label in LABELS:
            title, description = LABEL_INFO[label]
            with st.container(border=True):
                left, right = st.columns([1, 4])
                left.markdown(f"**{title}**")
                left.caption(f"`{label}`")
                right.write(description)

    else:
        st.markdown("## О проекте")
        st.markdown("### Задача")
        st.write("Определить намерение пользователя по последнему сообщению и доступной истории диалога. Это позволяет системе выбрать корректный маршрут обработки запроса.")
        st.markdown("### Почему учитывается история")
        st.write("Короткая фраза может иметь разные значения в зависимости от предыдущих сообщений. Поэтому модель анализирует не только текущую реплику, но и контекст.")
        st.markdown("### Архитектура дополнительного уточнения")
        st.write("Основной ансамбль сначала определяет один из восьми маршрутов по текущему сообщению и истории. Только если выбран `data_catalog`, запускается отдельный бинарный классификатор неоднозначности: GigaChat EmbeddingsGigaR проецируются PCA до 1 024 компонент и подаются в Logistic Regression; параллельно в артефакте сохранена TF-IDF-ветка по тексту и истории для дальнейших вариантов ансамбля. Предсказание `needs_clarification=true` означает не нехватку любого параметра, а строго ситуацию, когда непонятно — искать объект данных или объяснять тему; для остальных intent флаг принудительно равен `false`.")
        st.markdown("### Роль GigaEvo")
        st.write("GigaEvo применялся как автоматизированный экспериментатор: перебирал варианты программ отбора признаков и помогал сравнивать конфигурации моделей. В Streamlit запускается уже сохранённая итоговая модель — GigaEvo заново не запускается для каждого запроса.")
        st.markdown("### Онлайн-классификация")
        st.write("По умолчанию используется гибридная конфигурация v2. При необходимости в боковой панели можно переключиться на сохранённую локальную контекстную ветку.")
        st.caption("API-ключ, если он используется, не должен храниться в GitHub; в Streamlit Cloud его задают через Secrets.")

if __name__ == "__main__":
    main()
