import sys
import re
import csv
import random

import numpy as np
import pandas as pd

def training_file():
    parts = __file__.split("/")
    return "/".join(parts[:-1] + ["ml_challenge_dataset.csv"])

TRAIN_FILE = training_file()
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

print(load_training_data()[0][0])

