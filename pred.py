"""
ML Challenge - Predict Painting from Survey Data

Includes:
- 70/15/15 split
- Individual model accuracies
- Ensemble models (majority, weighted, top3)
- Final predictions
"""

import csv
import sys
import random
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# # =========================
# # LOAD + PREPROCESS
# # =========================

# df = pd.read_csv('ml_challenge_dataset.csv')
# df = df.dropna(subset=['Painting'])
# df = df.fillna('')

# feature_cols = [col for col in df.columns if col != 'Painting']
# X = df[feature_cols].astype(str).agg(' '.join, axis=1)
# y = df['Painting']


# # =========================
# # VECTORIZE + ENCODE
# # =========================

# vectorizer = TfidfVectorizer()
# X_vec = vectorizer.fit_transform(X)

# le = LabelEncoder()
# y_enc = le.fit_transform(y)


# # =========================
# # SPLIT (70 / 15 / 15)
# # =========================

# X_train, X_temp, y_train, y_temp = train_test_split(
#     X_vec, y_enc, test_size=0.30, random_state=42
# )

# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.50, random_state=42
# )


# # =========================
# # MODELS
# # =========================

# models = {
#     'mlp': MLPClassifier(random_state=42, max_iter=1000),
#     'rf': RandomForestClassifier(random_state=42),
#     'nb': MultinomialNB(),
#     'dt': DecisionTreeClassifier(random_state=42),
#     # 'knn': KNeighborsClassifier()
# }


# # =========================
# # ENSEMBLE FUNCTIONS
# # =========================

# def majority_vote(preds_dict):
#     df_preds = pd.DataFrame(preds_dict)
#     return df_preds.mode(axis=1)[0]


# def weighted_vote(preds_dict, weights):
#     df_preds = pd.DataFrame(preds_dict)
#     final = []

#     for _, row in df_preds.iterrows():
#         scores = {}
#         for model_name, pred in row.items():
#             scores[pred] = scores.get(pred, 0) + weights[model_name]
#         final.append(max(scores, key=scores.get))

#     return final


# def top3_vote(preds_dict, top_models):
#     df_preds = pd.DataFrame({k: v for k, v in preds_dict.items() if k in top_models})
#     return df_preds.mode(axis=1)[0]


# # =========================
# # TRAIN + EVALUATE
# # =========================

# print("\nModel Accuracies:")

# val_accuracies = {}
# test_predictions = {}
# val_predictions = {}

# for name, model in models.items():
#     model.fit(X_train, y_train)

#     # Train acc
#     train_acc = accuracy_score(y_train, model.predict(X_train))

#     # Val acc
#     val_preds = model.predict(X_val)
#     val_acc = accuracy_score(y_val, val_preds)

#     # Test acc
#     test_preds = model.predict(X_test)
#     test_acc = accuracy_score(y_test, test_preds)

#     print(f"{name}: train={train_acc:.4f}, val={val_acc:.4f}, test={test_acc:.4f}")

#     val_accuracies[name] = val_acc
#     val_predictions[name] = le.inverse_transform(val_preds)
#     test_predictions[name] = le.inverse_transform(test_preds)


# # =========================
# # ENSEMBLE EVALUATION
# # =========================

# # Majority Vote
# mv_val = majority_vote(val_predictions)
# mv_test = majority_vote(test_predictions)

# # Weighted Vote
# wv_val = weighted_vote(val_predictions, val_accuracies)
# wv_test = weighted_vote(test_predictions, val_accuracies)

# # Top 3 models (exclude weak ones)
# top_models = ['rf', 'nb', 'mlp']
# t3_val = top3_vote(val_predictions, top_models)
# t3_test = top3_vote(test_predictions, top_models)

# # Convert y back
# y_val_labels = le.inverse_transform(y_val)
# y_test_labels = le.inverse_transform(y_test)

# print("\nEnsemble Accuracies:")
# print(f"majority_vote: val={accuracy_score(y_val_labels, mv_val):.4f}, test={accuracy_score(y_test_labels, mv_test):.4f}")
# print(f"weighted_vote: val={accuracy_score(y_val_labels, wv_val):.4f}, test={accuracy_score(y_test_labels, wv_test):.4f}")
# print(f"top3_vote:     val={accuracy_score(y_val_labels, t3_val):.4f}, test={accuracy_score(y_test_labels, t3_test):.4f}")


# # =========================
# # RETRAIN ON FULL DATA
# # =========================

# for name, model in models.items():
#     model.fit(X_vec, y_enc)


# # =========================
# # PREDICT FUNCTION
# # =========================

# def predict_all(filename):
#     test_df = pd.read_csv(filename)
#     test_df = test_df.fillna('')

#     feature_cols = [col for col in test_df.columns if col != 'Painting']
#     test_X = test_df[feature_cols].astype(str).agg(' '.join, axis=1)
#     test_X_vec = vectorizer.transform(test_X)

#     all_preds = {}

#     for name, model in models.items():
#         preds = le.inverse_transform(model.predict(test_X_vec))
#         all_preds[name] = preds

#     df_preds = pd.DataFrame(all_preds)

#     # Add ensemble outputs
#     df_preds['majority_vote'] = df_preds.mode(axis=1)[0]
#     df_preds['top3_vote'] = df_preds[['rf', 'nb', 'mlp']].mode(axis=1)[0]

#     return df_preds


# =========================
# MAIN
# =========================

features = []
labels = []

# FALSE IS INCLUDED TRUE IS EXCLUDED
input_mask = [False, True, False, True, False, False, False, False, False, False, False, False, False, True, True, True]
# input_mask = np.array([not a for a in input_mask])

def sanitize(row):
    if len(row) != 16: return
    return np.array(row)[input_mask]
    # return np.ma.masked_array(row, mask=input_mask).compressed()
    # return np.array(row)
    
def to_bow(s):
    return [s]

def to_feature(row):
    label = []
    if row[0] == 'The Water Lily Pond':
        label = 0
    elif row[0] == 'The Starry Night':
        label = 1
    else:
        label = 2
    
    feature = []
    feature.extend(to_bow(row[1]))

    season = row[2].split(',')
    if 'Winter' in season:
        feature.append(1)
    else:
        feature.append(0)
    if 'Spring' in season:
        feature.append(1)
    else:
        feature.append(0)
    if 'Summer' in season:
        feature.append(1)
    else:
        feature.append(0)
    if 'Fall' in season:
        feature.append(1)
    else:
        feature.append(0)

    feature.extend(to_bow(row[3]))
    feature.extend(to_bow(row[4]))

    return feature, label

def extract(filename):
    """Reads the CSV and extracts a feature vector for each row."""
    data = csv.DictReader(open(filename, encoding='utf-8'))
    for i, r in enumerate(data):
        if i == 0: continue
        feature, label = to_feature(sanitize(list(r.values())))
        print(feature)
        if feature is not None:
            features.append(feature)
            labels.append(label)

def split_data(features, labels, frac_train, frac_valid, frac_test):
    assert np.equal(frac_train + frac_valid + frac_test, 1)

    paired = zip(features, labels)
    random.shuffle(paired)


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) == 2 else 'ml_challenge_dataset.csv'

    # 1. Fit BoW vocabulary from the dataset
    fit_bow('ml_challenge_dataset.csv')

    # 2. Extract features
    extract('ml_challenge_dataset.csv')
    print(features)

    split_data(features, labels, 0.7, 0.15, 0.15)

    # print("\nSample Predictions:")
    # print(predictions_df.head().to_string())

    # predictions_df.to_csv("predictions.csv", index=False)
    