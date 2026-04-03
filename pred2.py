"""
Restricted-import predictor for the ML challenge.

Allowed imports only:
- sys
- csv
- random
- numpy
- pandas

This file implements:
- text sanitization
- bag-of-words vocabulary creation
- lightweight structured feature tokens
- a manual multinomial Naive Bayes classifier
- predict_all(filename)
"""

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy
import pandas


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

MULTI_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
]

NUMERIC_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
]

PRICE_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"

PAINTINGS = [
    "The Water Lily Pond",
    "The Starry Night",
    "The Persistence of Memory",
]

PAINTING_TO_LABEL = {
    "The Water Lily Pond": 0,
    "The Starry Night": 1,
    "The Persistence of Memory": 2,
}

LIKERT_MAP = {
    "1 - Strongly disagree": "sd1",
    "2 - Disagree": "sd2",
    "3 - Neutral/Unsure": "sd3",
    "4 - Agree": "sd4",
    "5 - Strongly agree": "sd5",
}

PUNCT_TRANSLATION = str.maketrans({
    ".": " ",
    ",": " ",
    "!": " ",
    "?": " ",
    ":": " ",
    ";": " ",
    "(": " ",
    ")": " ",
    "[": " ",
    "]": " ",
    "{": " ",
    "}": " ",
    "\"": " ",
    "'": " ",
    "/": " ",
    "\\": " ",
    "-": " ",
    "_": " ",
    "$": " ",
    "&": " ",
    "*": " ",
    "+": " ",
    "=": " ",
    "@": " ",
    "#": " ",
    "%": " ",
    "^": " ",
    "~": " ",
    "`": " ",
    "|": " ",
    "<": " ",
    ">": " ",
})

MAX_VOCAB_SIZE = 5000
SMOOTHING = 1.0

MODEL = None


def training_filename():
    parts = __file__.split("/")
    return "/".join(parts[:-1] + ["ml_challenge_dataset.csv"])


def clamp(value, low, high):
    return max(low, min(high, value))


def sanitize_text(text):
    text = str(text).lower().strip()
    replacements = [
        ("this painting makes me feel", " "),
        ("this art piece makes me feel", " "),
        ("it makes me feel", " "),
        ("makes me feel", " "),
        ("i feel", " "),
        ("ice cream", "ice_cream"),
        ("apple pie", "apple_pie"),
        ("blueberry pie", "blueberry_pie"),
        ("living room", "living_room"),
        ("dining room", "dining_room"),
        ("family members", "family_members"),
        ("by yourself", "by_yourself"),
        ("coworkers/classmates", "coworkers_classmates"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = text.translate(PUNCT_TRANSLATION)
    text = text.replace("noodles", "noodle")
    text = text.replace("fries", "fry")
    text = text.replace("potatoes", "potato")
    text = text.replace("berries", "berry")
    return " ".join(text.split())


def tokenize(text):
    cleaned = sanitize_text(text)
    if not cleaned:
        return []
    return cleaned.split()


def split_multi(value):
    items = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            items.append(item)
    return items


def parse_float(value):
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_price(value):
    text = str(value).strip().lower().replace(",", "")
    if text == "":
        return None

    digits = []
    current = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            current += ch
        elif current != "":
            digits.append(current)
            current = ""
    if current != "":
        digits.append(current)

    if not digits:
        return None

    try:
        return float(digits[0])
    except ValueError:
        return None


def price_bucket(price):
    if price is None:
        return "missing"
    if price == 0:
        return "0"
    if price <= 50:
        return "1_50"
    if price <= 200:
        return "51_200"
    if price <= 500:
        return "201_500"
    if price <= 1000:
        return "501_1000"
    return "1000_plus"


def numeric_bucket(name, value):
    if value is None:
        return name + "_missing"

    if name == "intensity":
        return name + "_" + str(int(clamp(value, 0, 10)))
    if name == "colours":
        value = clamp(value, 0, 15)
        if value <= 2:
            return "colours_low"
        if value <= 5:
            return "colours_mid"
        return "colours_high"
    if name == "objects":
        value = clamp(value, 0, 15)
        if value <= 2:
            return "objects_low"
        if value <= 5:
            return "objects_mid"
        return "objects_high"
    return name + "_missing"


def row_tokens(row):
    tokens = []

    for col in TEXT_COLS:
        tokens.extend(tokenize(row.get(col, "")))

    intensity = parse_float(row.get(NUMERIC_COLS[0], ""))
    colours = parse_float(row.get(NUMERIC_COLS[1], ""))
    objects = parse_float(row.get(NUMERIC_COLS[2], ""))

    tokens.append("num_" + numeric_bucket("intensity", intensity))
    tokens.append("num_" + numeric_bucket("colours", colours))
    tokens.append("num_" + numeric_bucket("objects", objects))

    price = parse_price(row.get(PRICE_COL, ""))
    tokens.append("price_" + price_bucket(price))

    for col in LIKERT_COLS:
        value = str(row.get(col, "")).strip()
        tokens.append("likert_" + LIKERT_MAP.get(value, "missing"))

    multi_values = {}
    for col in MULTI_COLS:
        values = split_multi(row.get(col, ""))
        multi_values[col] = values
        if not values:
            tokens.append("multi_missing_" + col)
        for value in values:
            tokens.append(col + "=" + sanitize_text(value).replace(" ", "_"))

    seasons = multi_values[MULTI_COLS[2]]
    rooms = multi_values[MULTI_COLS[0]]
    viewing = multi_values[MULTI_COLS[1]]

    for season in seasons:
        season_key = sanitize_text(season).replace(" ", "_")
        for room in rooms:
            room_key = sanitize_text(room).replace(" ", "_")
            tokens.append("season_room=" + season_key + "|" + room_key)
        for who in viewing:
            who_key = sanitize_text(who).replace(" ", "_")
            tokens.append("season_view=" + season_key + "|" + who_key)

    for room in rooms:
        room_key = sanitize_text(room).replace(" ", "_")
        for who in viewing:
            who_key = sanitize_text(who).replace(" ", "_")
            tokens.append("room_view=" + room_key + "|" + who_key)

    if intensity is not None and colours is not None:
        tokens.append("ixc_" + str(int(clamp(intensity * colours, 0, 100) // 10)))
    if intensity is not None and objects is not None:
        tokens.append("ixo_" + str(int(clamp(intensity * objects, 0, 100) // 10)))
    if colours is not None and objects is not None:
        tokens.append("cxo_" + str(int(clamp(colours * objects, 0, 100) // 10)))

    return tokens


def sanitize(row):
    return row_tokens(row)


def build_vocab(rows):
    counts = {}
    for row in rows:
        for token in row_tokens(row):
            counts[token] = counts.get(token, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    vocab = [token for token, _ in sorted_items[:MAX_VOCAB_SIZE]]
    vocab_index = {}
    for i, token in enumerate(vocab):
        vocab_index[token] = i
    return vocab, vocab_index


def to_BoW(row, vocab_index):
    x = numpy.zeros(len(vocab_index), dtype=float)
    for token in row_tokens(row):
        if token in vocab_index:
            x[vocab_index[token]] += 1.0
    return x


def rows_to_matrix(rows, vocab_index):
    X = numpy.zeros((len(rows), len(vocab_index)), dtype=float)
    for i, row in enumerate(rows):
        X[i] = to_BoW(row, vocab_index)
    return X


def train_naive_bayes(X, y):
    num_classes = len(PAINTINGS)
    vocab_size = X.shape[1]

    class_counts = numpy.zeros(num_classes, dtype=float)
    token_counts = numpy.zeros((num_classes, vocab_size), dtype=float)

    for label in range(num_classes):
        class_mask = (y == label)
        class_counts[label] = numpy.sum(class_mask)
        if class_counts[label] > 0:
            token_counts[label] = X[class_mask].sum(axis=0)

    priors = (class_counts + SMOOTHING) / (numpy.sum(class_counts) + SMOOTHING * num_classes)
    likelihoods = (
        token_counts + SMOOTHING
    ) / (
        token_counts.sum(axis=1, keepdims=True) + SMOOTHING * vocab_size
    )

    return {
        "priors": numpy.log(priors),
        "log_likelihoods": numpy.log(likelihoods),
    }

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_neural_net(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y) if y.dtype == object else y

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,          # L2 regularization (replaces your SMOOTHING role)
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.3,
        random_state=42,
        verbose=False,
    )

    model.fit(X, y_encoded)
    return model


def predict_neural_net(model, X):
    return model.predict(X)


def predict_proba_neural_net(model, X):
    return model.predict_proba(X)

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',   # standard for classification
        random_state=42,
        n_jobs=-1,             # use all cores
    )
    model.fit(X, y)
    return model

def predict_matrix(model, X):
    scores = X.dot(model["log_likelihoods"].T) + model["priors"]
    return numpy.argmax(scores, axis=1)


def load_rows(filename):
    data = pandas.read_csv(filename)
    data = data.fillna("")
    return data.to_dict(orient="records")


def fit_model():
    rows = load_rows(training_filename())
    train_rows = []
    labels = []

    for row in rows:
        painting = row.get(LABEL_COL, "")
        if painting in PAINTING_TO_LABEL:
            train_rows.append(row)
            labels.append(PAINTING_TO_LABEL[painting])

    y = numpy.array(labels, dtype=int)
    vocab, vocab_index = build_vocab(train_rows)
    X = rows_to_matrix(train_rows, vocab_index)
    model = train_naive_bayes(X, y)
    model2 = train_neural_net(X, y)
    model3 = train_random_forest(X, y)

    return {
        "vocab": vocab,
        "vocab_index": vocab_index,
        "model": model,
        "model2": model2,
        "model3": model3,
    }


def fit_model_from_rows(train_rows, labels):
    y = numpy.array(labels, dtype=int)
    vocab, vocab_index = build_vocab(train_rows)
    X = rows_to_matrix(train_rows, vocab_index)
    model = train_naive_bayes(X, y)
    model2 = train_neural_net(X, y)
    model3 = train_random_forest(X, y)
    return {
        "vocab": vocab,
        "vocab_index": vocab_index,
        "model": model,
        "model2": model2,
        "model3": model3,
    }


def ensure_model():
    global MODEL
    if MODEL is None:
        MODEL = fit_model()
    return MODEL


def predict(x):
    state = ensure_model()
    X = numpy.zeros((1, len(state["vocab_index"])), dtype=float)
    X[0] = to_BoW(x, state["vocab_index"])
    
    pred = state["model2"].predict(X)[0]
    return PAINTINGS[int(pred)]
    pred = predict_matrix(state["model"], X)[0]
    return PAINTINGS[int(pred)]



def accuracy_for_rows(state, rows, labels):
    X = rows_to_matrix(rows, state["vocab_index"])
    preds = predict_matrix(state["model"], X)
    preds = state["model2"].predict(X)
    labels = numpy.array(labels, dtype=int)
    return float(numpy.mean(preds == labels))


def evaluate_model():
    rows = load_rows(training_filename())
    grouped = {}
    for painting in PAINTINGS:
        grouped[painting] = []

    for row in rows:
        painting = row.get(LABEL_COL, "")
        if painting in PAINTING_TO_LABEL:
            grouped[painting].append((row, PAINTING_TO_LABEL[painting]))

    train_pairs = []
    val_pairs = []
    test_pairs = []

    for painting in PAINTINGS:
        paired = grouped[painting]
        random.shuffle(paired)

        n = len(paired)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train_pairs.extend(paired[:n_train])
        val_pairs.extend(paired[n_train:n_train + n_val])
        test_pairs.extend(paired[n_train + n_val:])

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)

    train_rows = [row for row, _ in train_pairs]
    train_labels = [label for _, label in train_pairs]
    val_rows = [row for row, _ in val_pairs]
    val_labels = [label for _, label in val_pairs]
    test_rows = [row for row, _ in test_pairs]
    test_labels = [label for _, label in test_pairs]

    state = fit_model_from_rows(train_rows, train_labels)

    print(state["model2"].coefs_)
    print(state["model2"].intercepts_)


    train_acc = accuracy_for_rows(state, train_rows, train_labels)
    val_acc = accuracy_for_rows(state, val_rows, val_labels)
    test_acc = accuracy_for_rows(state, test_rows, test_labels)

    return len(state["vocab"]), train_acc, val_acc, test_acc


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    ensure_model()
    data = load_rows(filename)

    predictions = []
    for test_example in data:
        pred = predict(test_example)
        predictions.append(pred)

    return predictions


if __name__ == "__main__":
    vocab_size, train_acc, val_acc, test_acc = evaluate_model()
    print("Vocabulary size:", vocab_size)
    print(
        "Naive Bayes accuracy:",
        "train={:.4f},".format(train_acc),
        "val={:.4f},".format(val_acc),
        "test={:.4f}".format(test_acc),
    )
    # test_file = sys.argv[1] if len(sys.argv) == 2 else training_filename()
    # preds = predict_all(test_file)
    # print("Generated", len(preds), "predictions.")
    # print(preds[:10])
