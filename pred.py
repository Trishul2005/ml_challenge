"""
Inference script for the CSC311 ML challenge.

This file exposes `predict_all(filename)`, which loads the exported random
forest from `predgpt_forest.npz`, rebuilds the same structured + one-hot text
features used by `predgpt.py`, and returns painting-name predictions.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

MODEL_FILE = Path(__file__).with_name("final_forest.npz")

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

TOKEN_RE = re.compile(r"(?u)\b\w[\w']+\b")

_MODEL_CACHE = None
bag = []


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
    return " ".join(str(row.get(col, "")).strip() for col in TEXT_COLS if str(row.get(col, "")).strip())


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

    intensity = max(features["intensity"], 0.0)
    colours = max(features["colours"], 0.0)
    objects = max(features["objects"], 0.0)
    features["intensity_x_colours"] = intensity * colours
    features["intensity_x_objects"] = intensity * objects
    features["colours_x_objects"] = colours * objects

    return features


def tokenize_with_bigrams(text):
    text = "" if text is None else str(text).lower()
    base_tokens = TOKEN_RE.findall(text)
    tokens = list(base_tokens)
    for i in range(len(base_tokens) - 1):
        tokens.append(base_tokens[i] + " " + base_tokens[i + 1])
    return tokens


def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    data = np.load(MODEL_FILE, allow_pickle=True)
    structured_names = [str(name) for name in data["structured_feature_names"]]
    text_names = [str(name) for name in data["text_feature_names"]]
    text_vocab = {token: i for i, token in enumerate(text_names)}
    n_estimators = int(data["n_estimators"][0])

    trees = []
    for i in range(n_estimators):
        prefix = f"tree_{i:03d}"
        trees.append(
            {
                "children_left": data[f"{prefix}_children_left"],
                "children_right": data[f"{prefix}_children_right"],
                "feature": data[f"{prefix}_feature"],
                "threshold": data[f"{prefix}_threshold"],
                "value": data[f"{prefix}_value"],
            }
        )

    _MODEL_CACHE = {
        "labels": [str(label) for label in data["label_names"]],
        "structured_names": structured_names,
        "text_vocab": text_vocab,
        "n_features": len(structured_names) + len(text_names),
        "trees": trees,
    }
    return _MODEL_CACHE


def vectorize_row(row, model):
    x = np.zeros(model["n_features"], dtype=np.float32)

    structured = row_to_structured_features(row)
    for i, name in enumerate(model["structured_names"]):
        x[i] = float(structured.get(name, 0.0))

    text_offset = len(model["structured_names"])
    for token in tokenize_with_bigrams(row_to_text(row)):
        j = model["text_vocab"].get(token)
        if j is not None:
            x[text_offset + j] += 1.0

    return x


def sanitize(row):
    """
    Preserve the original starter-file function name.
    Returns the cleaned representation used for feature building.
    """
    return {
        "structured": row_to_structured_features(row),
        "text": row_to_text(row),
    }


def to_BoW(row):
    """
    Preserve the original starter-file function name.
    Converts one input row into the one-hot / bag-of-words vector expected by
    the exported forest model.
    """
    model = load_model()
    return vectorize_row(row, model)


def predict_tree(tree, x):
    node = 0
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    features = tree["feature"]
    thresholds = tree["threshold"]
    values = tree["value"]

    while children_left[node] != children_right[node]:
        feature_idx = features[node]
        if x[feature_idx] <= thresholds[node]:
            node = children_left[node]
        else:
            node = children_right[node]

    return values[node]


def predict_row(row, model):
    x = vectorize_row(row, model)
    scores = np.zeros(len(model["labels"]), dtype=np.float64)

    for tree in model["trees"]:
        scores += predict_tree(tree, x)

    return model["labels"][int(np.argmax(scores))]


def predict(x):
    """
    Preserve the original starter-file function name.
    """
    model = load_model()
    return predict_row(x, model)


def predict_all(filename):
    """
    Make predictions for the data in filename.
    """
    with open(filename, encoding="utf-8") as f:
        data = csv.DictReader(f)
        return [predict(row) for row in data]


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "testing.csv"
    predictions = predict_all(input_file)
    for prediction in predictions:
        print(prediction)
