"""
Search sklearn Naive Bayes configurations for the ML challenge dataset.

This version uses the same engineered token idea as prednaive.py so the
sklearn search is tuning the right feature space instead of a weaker raw-text one.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB


TRAIN_FILE = Path(__file__).with_name("ml_challenge_dataset.csv")
LABEL_COL = "Painting"
RANDOM_STATE = 42

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

LIKERT_MAP = {
    "1 - Strongly disagree": "sd1",
    "2 - Disagree": "sd2",
    "3 - Neutral/Unsure": "sd3",
    "4 - Agree": "sd4",
    "5 - Strongly agree": "sd5",
}

PUNCT_TRANSLATION = str.maketrans({
    ".": " ", ",": " ", "!": " ", "?": " ", ":": " ", ";": " ",
    "(": " ", ")": " ", "[": " ", "]": " ", "{": " ", "}": " ",
    "\"": " ", "'": " ", "/": " ", "\\": " ", "-": " ", "_": " ",
    "$": " ", "&": " ", "*": " ", "+": " ", "=": " ", "@": " ",
    "#": " ", "%": " ", "^": " ", "~": " ", "`": " ", "|": " ",
    "<": " ", ">": " ",
})


def load_data(filename=TRAIN_FILE):
    df = pd.read_csv(filename).fillna("")
    return df[df[LABEL_COL].isin([
        "The Water Lily Pond",
        "The Starry Night",
        "The Persistence of Memory",
    ])].copy()


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
    return [item.strip() for item in str(value).split(",") if item.strip()]


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
        elif current:
            digits.append(current)
            current = ""
    if current:
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
        tokens.extend(tokenize(row[col]))

    intensity = parse_float(row[NUMERIC_COLS[0]])
    colours = parse_float(row[NUMERIC_COLS[1]])
    objects = parse_float(row[NUMERIC_COLS[2]])

    tokens.append("num_" + numeric_bucket("intensity", intensity))
    tokens.append("num_" + numeric_bucket("colours", colours))
    tokens.append("num_" + numeric_bucket("objects", objects))
    tokens.append("price_" + price_bucket(parse_price(row[PRICE_COL])))

    for col in LIKERT_COLS:
        tokens.append("likert_" + LIKERT_MAP.get(str(row[col]).strip(), "missing"))

    multi_values = {}
    for col in MULTI_COLS:
        values = split_multi(row[col])
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


def row_to_engineered_text(row):
    return " ".join(row_tokens(row))


def evaluate_pipeline(name, vectorizer, model, X_train_text, X_val_text, X_test_text, y_train, y_val, y_test):
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    model.fit(X_train, y_train)

    return {
        "name": name,
        "train": accuracy_score(y_train, model.predict(X_train)),
        "val": accuracy_score(y_val, model.predict(X_val)),
        "test": accuracy_score(y_test, model.predict(X_test)),
        "features": X_train.shape[1],
    }


def build_experiments():
    exps = []

    for alpha in [1.0, 0.5, 0.3, 0.1, 0.03]:
        exps.append((
            f"multinomial_count_uni_bi_8k_min1_a{alpha}",
            CountVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1),
            MultinomialNB(alpha=alpha),
        ))
        exps.append((
            f"multinomial_count_uni_bi_12k_min1_a{alpha}",
            CountVectorizer(max_features=12000, ngram_range=(1, 2), min_df=1),
            MultinomialNB(alpha=alpha),
        ))
        exps.append((
            f"multinomial_count_uni_bi_12k_min2_a{alpha}",
            CountVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2),
            MultinomialNB(alpha=alpha),
        ))
        exps.append((
            f"multinomial_tfidf_uni_bi_12k_min1_a{alpha}",
            TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=1, use_idf=True, sublinear_tf=False),
            MultinomialNB(alpha=alpha),
        ))
        exps.append((
            f"multinomial_tfidf_sublinear_uni_bi_12k_min1_a{alpha}",
            TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=1, use_idf=True, sublinear_tf=True),
            MultinomialNB(alpha=alpha),
        ))
        exps.append((
            f"complement_count_uni_bi_12k_min1_a{alpha}",
            CountVectorizer(max_features=12000, ngram_range=(1, 2), min_df=1),
            ComplementNB(alpha=alpha),
        ))
        exps.append((
            f"complement_tfidf_uni_bi_12k_min1_a{alpha}",
            TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            ComplementNB(alpha=alpha),
        ))
        exps.append((
            f"bernoulli_binary_uni_bi_12k_min1_a{alpha}",
            CountVectorizer(max_features=12000, ngram_range=(1, 2), min_df=1, binary=True),
            BernoulliNB(alpha=alpha),
        ))

    # A few bigger / different ngram settings.
    exps.extend([
        (
            "multinomial_count_uni_tri_15k_min1_a0.3",
            CountVectorizer(max_features=15000, ngram_range=(1, 3), min_df=1),
            MultinomialNB(alpha=0.3),
        ),
        (
            "multinomial_tfidf_uni_tri_15k_min1_a0.3",
            TfidfVectorizer(max_features=15000, ngram_range=(1, 3), min_df=1, sublinear_tf=True),
            MultinomialNB(alpha=0.3),
        ),
        (
            "bernoulli_binary_uni_tri_15k_min1_a0.3",
            CountVectorizer(max_features=15000, ngram_range=(1, 3), min_df=1, binary=True),
            BernoulliNB(alpha=0.3),
        ),
    ])

    return exps


def main():
    df = load_data()
    X_text = df.apply(row_to_engineered_text, axis=1)
    y = df[LABEL_COL]

    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X_text,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    results = []
    for name, vectorizer, model in build_experiments():
        result = evaluate_pipeline(
            name,
            vectorizer,
            model,
            X_train_text,
            X_val_text,
            X_test_text,
            y_train,
            y_val,
            y_test,
        )
        results.append(result)
        print(
            f"{name}: "
            f"train={result['train']:.4f}, "
            f"val={result['val']:.4f}, "
            f"test={result['test']:.4f}, "
            f"features={result['features']}"
        )

    results.sort(key=lambda item: (item["val"], item["test"]), reverse=True)
    best = results[0]

    print("\nTop 5 by validation accuracy:")
    for result in results[:5]:
        print(
            f"{result['name']}: "
            f"train={result['train']:.4f}, "
            f"val={result['val']:.4f}, "
            f"test={result['test']:.4f}, "
            f"features={result['features']}"
        )

    print("\nBest by validation accuracy:")
    print(
        f"{best['name']}: "
        f"train={best['train']:.4f}, "
        f"val={best['val']:.4f}, "
        f"test={best['test']:.4f}, "
        f"features={best['features']}"
    )


if __name__ == "__main__":
    main()
