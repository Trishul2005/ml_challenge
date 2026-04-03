"""
Restricted-import wrapper around the trained sklearn random forest bundle.

This file only imports:
- sys
- csv
- random
- numpy
- pandas

The actual vectorizers and trained forest are loaded from predreal.p.
"""

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy
import pandas


MODEL_BUNDLE = None
PAINTING_TO_LABEL = {
    "The Water Lily Pond": 0,
    "The Starry Night": 1,
    "The Persistence of Memory": 2,
}


def bundle_filename():
    parts = __file__.split("/")
    return "/".join(parts[:-1] + ["predreal.p"])


def data_filename():
    parts = __file__.split("/")
    return "/".join(parts[:-1] + ["ml_challenge_dataset.csv"])


def ensure_bundle():
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        MODEL_BUNDLE = pandas.read_pickle(bundle_filename())
    return MODEL_BUNDLE


def load_rows(filename):
    data = pandas.read_csv(filename)
    data = data.fillna("")
    return data.to_dict(orient="records")


def row_to_structured_features(row):
    price_col = "How much (in Canadian dollars) would you be willing to pay for this painting?"
    numeric_cols = {
        "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": ("intensity", 0.0, 10.0),
        "How many prominent colours do you notice in this painting?": ("colours", 0.0, 15.0),
        "How many objects caught your eye in the painting?": ("objects", 0.0, 15.0),
    }
    likert_cols = [
        "This art piece makes me feel sombre.",
        "This art piece makes me feel content.",
        "This art piece makes me feel calm.",
        "This art piece makes me feel uneasy.",
    ]
    likert_map = {
        "1 - Strongly disagree": 1.0,
        "2 - Disagree": 2.0,
        "3 - Neutral/Unsure": 3.0,
        "4 - Agree": 4.0,
        "5 - Strongly agree": 5.0,
    }
    multi_choice_cols = [
        "If you could purchase this painting, which room would you put that painting in?",
        "If you could view this art in person, who would you want to view it with?",
        "What season does this art piece remind you of?",
    ]

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
        text = str(value).strip().lower().replace(",", "")
        if not text:
            return None
        pieces = []
        current = ""
        for ch in text:
            if ch.isdigit() or ch == ".":
                current += ch
            elif current:
                pieces.append(current)
                current = ""
        if current:
            pieces.append(current)
        if not pieces:
            return None
        try:
            return float(pieces[0])
        except ValueError:
            return None

    def split_multi_value(value):
        return [item.strip() for item in str(value).split(",") if item.strip()]

    features = {}

    for column, values in numeric_cols.items():
        name, low, high = values
        value = parse_float(row.get(column, ""))
        if value is None:
            features[name] = -1.0
            features[name + "_missing"] = 1.0
        else:
            features[name] = clamp(value, low, high)
            features[name + "_missing"] = 0.0

    price = parse_price(row.get(price_col, ""))
    if price is None:
        features["price"] = -1.0
        features["log_price"] = 0.0
        features["price_missing"] = 1.0
    else:
        price = clamp(price, 0.0, 5000.0)
        features["price"] = price
        features["log_price"] = float(numpy.log1p(price))
        features["price_missing"] = 0.0

    for column in likert_cols:
        value = str(row.get(column, "")).strip()
        features[column] = likert_map.get(value, -1.0)
        features[column + "_missing"] = 0.0 if value else 1.0

    for column in multi_choice_cols:
        values = split_multi_value(row.get(column, ""))
        features[column + "__count"] = float(len(values))
        features[column + "__missing"] = 0.0 if values else 1.0
        for item in values:
            features[column + "=" + item] = 1.0

    intensity = max(features["intensity"], 0.0)
    colours = max(features["colours"], 0.0)
    objects = max(features["objects"], 0.0)
    features["intensity_x_colours"] = intensity * colours
    features["intensity_x_objects"] = intensity * objects
    features["colours_x_objects"] = colours * objects

    return features


def row_to_text(row):
    text_cols = [
        "Describe how this painting makes you feel.",
        "If this painting was a food, what would be?",
        "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
    ]
    parts = []
    for col in text_cols:
        value = str(row.get(col, "")).strip()
        if value:
            parts.append(value)
    return " ".join(parts)


def transform_rows(rows, bundle):
    structured = bundle["full_structured_vectorizer"].transform(
        [row_to_structured_features(row) for row in rows]
    )
    text = bundle["full_text_vectorizer"].transform([row_to_text(row) for row in rows])
    return numpy.hstack([structured, text.toarray()])


def predict(row):
    bundle = ensure_bundle()
    X = transform_rows([row], bundle)
    pred = bundle["full_model"].predict(X)[0]
    return bundle["label_to_painting"][int(pred)]


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    bundle = ensure_bundle()
    rows = load_rows(filename)
    X = transform_rows(rows, bundle)
    preds = bundle["full_model"].predict(X)
    return [bundle["label_to_painting"][int(pred)] for pred in preds]


if __name__ == "__main__":
    bundle = ensure_bundle()
    test_file = sys.argv[1] if len(sys.argv) == 2 else data_filename()
    metrics = bundle.get("eval_metrics")
    if metrics is not None:
        print(
            "Random forest accuracy:",
            "train={:.4f},".format(metrics["train"]),
            "val={:.4f},".format(metrics["val"]),
            "test={:.4f}".format(metrics["test"]),
        )
    preds = predict_all(test_file)
    print("Generated", len(preds), "predictions.")
    print(preds[:10])
