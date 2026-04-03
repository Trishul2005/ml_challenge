"""
Random forest baseline for the ML challenge.

This version uses a hybrid feature pipeline:
- structured survey features parsed into numeric / multi-hot features
- bag-of-words text features from the free-response columns
- random forest training with a 70/15/15 evaluation split
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


TRAIN_FILE = Path(__file__).with_name("ml_challenge_dataset.csv")
LABEL_COL = "Painting"
ID_COL = "unique_id"

TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]

LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]

MULTI_CHOICE_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
]

NUMERIC_COLS = {
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": ("intensity", 0.0, 10.0),
    "How many prominent colours do you notice in this painting?": ("colours", 0.0, 15.0),
    "How many objects caught your eye in the painting?": ("objects", 0.0, 15.0),
}

PRICE_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"

LIKERT_MAP = {
    "1 - Strongly disagree": 1.0,
    "2 - Disagree": 2.0,
    "3 - Neutral/Unsure": 3.0,
    "4 - Agree": 4.0,
    "5 - Strongly agree": 5.0,
}

PAINTING_TO_LABEL = {
    "The Water Lily Pond": 0,
    "The Starry Night": 1,
    "The Persistence of Memory": 2,
}

LABEL_TO_PAINTING = {value: key for key, value in PAINTING_TO_LABEL.items()}


class DictVectorizer:
    def __init__(self, sparse):
        self.sparse = sparse
        self.feature_names_ = []
        self.vocabulary_ = {}

    def fit(self, rows):
        keys = set()
        for row in rows:
            keys.update(row.keys())
        self.feature_names_ = sorted(keys)
        self.vocabulary_ = {name: i for i, name in enumerate(self.feature_names_)}
        return self

    def transform(self, rows):
        X = np.zeros((len(rows), len(self.feature_names_)), dtype=np.float32)
        for i, row in enumerate(rows):
            for key, value in row.items():
                j = self.vocabulary_.get(key)
                if j is not None:
                    X[i, j] = float(value)
        return X

    def fit_transform(self, rows):
        self.fit(rows)
        return self.transform(rows)

    def get_feature_names_out(self):
        return np.array(self.feature_names_, dtype=object)


class CountVectorizer:
    def __init__(
        self,
        lowercase,
        token_pattern,
        ngram_range,
        min_df,
        max_features
    ):
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_ = {}
        self._feature_names = []
        self._token_re = re.compile(token_pattern)

    def _tokenize(self, text):
        text = "" if text is None else str(text)
        if self.lowercase:
            text = text.lower()
        base_tokens = self._token_re.findall(text)

        min_n, max_n = self.ngram_range
        all_tokens = []
        for n in range(min_n, max_n + 1):
            if n <= 0:
                continue
            if n == 1:
                all_tokens.extend(base_tokens)
                continue
            if len(base_tokens) < n:
                continue
            for i in range(len(base_tokens) - n + 1):
                all_tokens.append(" ".join(base_tokens[i : i + n]))
        return all_tokens

    def fit(self, texts):
        doc_freq = {}
        n_docs = len(texts)

        for text in texts:
            seen = set(self._tokenize(text))
            for token in seen:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        if isinstance(self.min_df, float):
            min_df_count = int(np.ceil(self.min_df * n_docs))
        else:
            min_df_count = int(self.min_df)
        min_df_count = max(1, min_df_count)

        items = [(tok, df) for tok, df in doc_freq.items() if df >= min_df_count]
        items.sort(key=lambda x: (-x[1], x[0]))
        if self.max_features is not None:
            items = items[: int(self.max_features)]

        self._feature_names = [tok for tok, _ in items]
        self.vocabulary_ = {tok: i for i, tok in enumerate(self._feature_names)}
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), len(self._feature_names)), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in self._tokenize(text):
                j = self.vocabulary_.get(token)
                if j is not None:
                    X[i, j] += 1.0
        return X

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names_out(self):
        return np.array(self._feature_names, dtype=object)

def clamp(value, low, high):
    return max(low, min(high, value))


def parse_float(value):
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_price(value):
    value = str(value).strip().lower().replace(",", "")
    if not value:
        return None
    matches = re.findall(r"\d+(?:\.\d+)?", value)
    if not matches:
        return None
    return float(matches[0])


def split_multi_value(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def row_to_text(row):
    return " ".join(str(row[col]).strip() for col in TEXT_COLS if str(row[col]).strip())


def row_to_structured_features(row):
    features = {}

    for column, (name, low, high) in NUMERIC_COLS.items():
        value = parse_float(row.get(column, ""))
        if value is None:
            features[name] = -1.0
            features[f"{name}_missing"] = 1.0
        else:
            features[name] = clamp(value, low, high)
            features[f"{name}_missing"] = 0.0

    price = parse_price(row.get(PRICE_COL, ""))
    if price is None:
        features["price"] = -1.0
        features["log_price"] = 0.0
        features["price_missing"] = 1.0
    else:
        price = clamp(price, 0.0, 5000.0)
        features["price"] = price
        features["log_price"] = np.log1p(price)
        features["price_missing"] = 0.0

    for column in LIKERT_COLS:
        value = str(row.get(column, "")).strip()
        features[column] = LIKERT_MAP.get(value, -1.0)
        features[f"{column}_missing"] = 0.0 if value else 1.0

    for column in MULTI_CHOICE_COLS:
        values = split_multi_value(row.get(column, ""))
        features[f"{column}__count"] = float(len(values))
        features[f"{column}__missing"] = 0.0 if values else 1.0
        for item in values:
            features[f"{column}={item}"] = 1.0

    # Let the forest split on simple nonlinear combinations too.
    intensity = max(features["intensity"], 0.0)
    colours = max(features["colours"], 0.0)
    objects = max(features["objects"], 0.0)
    features["intensity_x_colours"] = intensity * colours
    features["intensity_x_objects"] = intensity * objects
    features["colours_x_objects"] = colours * objects

    return features


def load_rows(filename):
    with open(filename, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_training_data(filename=TRAIN_FILE):
    rows = [row for row in load_rows(filename) if row.get(LABEL_COL) in PAINTING_TO_LABEL]
    labels = np.array([PAINTING_TO_LABEL[row[LABEL_COL]] for row in rows], dtype=np.int64)
    return rows, labels


def build_feature_matrices(train_rows, other_rows):
    structured_vectorizer = DictVectorizer(sparse=False)
    text_vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w[\w']+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=8000,
    )

    train_struct = structured_vectorizer.fit_transform(
        [row_to_structured_features(row) for row in train_rows]
    )
    train_text = text_vectorizer.fit_transform([row_to_text(row) for row in train_rows])
    train_X = np.hstack([train_struct, train_text])

    other_matrices = []
    for rows in other_rows:
        struct = structured_vectorizer.transform([row_to_structured_features(row) for row in rows])
        text = text_vectorizer.transform([row_to_text(row) for row in rows])
        other_matrices.append(np.hstack([struct, text]))

    return train_X, other_matrices, structured_vectorizer, text_vectorizer


def transform_rows(rows, structured_vectorizer, text_vectorizer):
    struct = structured_vectorizer.transform([row_to_structured_features(row) for row in rows])
    text = text_vectorizer.transform([row_to_text(row) for row in rows])
    return np.hstack([struct, text])


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=900,
        class_weight="balanced_subsample",
        max_features="sqrt",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_all(filename):
    train_rows, train_labels = load_training_data()
    X_train, _, structured_vectorizer, text_vectorizer = build_feature_matrices(train_rows, [])
    model = train_model(X_train, train_labels)

    test_rows = load_rows(filename)
    X_test = transform_rows(test_rows, structured_vectorizer, text_vectorizer)
    preds = model.predict(X_test)
    return [LABEL_TO_PAINTING[int(pred)] for pred in preds]


def main():
    all_rows, all_labels = load_training_data()

    rows_train, rows_temp, y_train, y_temp = train_test_split(
        all_rows,
        all_labels,
        test_size=0.30,
        stratify=all_labels,
    )

    rows_val, rows_test, y_val, y_test = train_test_split(
        rows_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
    )

    X_train, [X_val, X_test], structured_vectorizer, text_vectorizer = build_feature_matrices(
        rows_train,
        [rows_val, rows_test],
    )

    model = train_model(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    vocab_size = len(text_vectorizer.vocabulary_)
    total_features = X_train.shape[1]

    print(f"Vocabulary size: {vocab_size}")
    print(f"Total feature count: {total_features}")
    print(
        "Random forest accuracy: "
        f"train={train_acc:.4f}, val={val_acc:.4f}, test={test_acc:.4f}"
    )

    # Retrain on the full labeled dataset before generating predictions.
    X_full, _, structured_vectorizer, text_vectorizer = build_feature_matrices(all_rows, [])
    final_model = train_model(X_full, all_labels)

    test_file = Path(sys.argv[1]) if len(sys.argv) == 2 else TRAIN_FILE
    prediction_rows = load_rows(test_file)
    X_pred = transform_rows(prediction_rows, structured_vectorizer, text_vectorizer)
    predictions = [LABEL_TO_PAINTING[int(pred)] for pred in final_model.predict(X_pred)]

    print(f"Generated {len(predictions)} predictions.")
    print(predictions[:10])


if __name__ == "__main__":
    main()
