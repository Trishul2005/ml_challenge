
import numpy as np
import predgpt as m

model, structured_vectorizer, text_vectorizer = m.get_best()

structured_feature_names = [None] * len(structured_vectorizer.vocabulary_)
for name, idx in structured_vectorizer.vocabulary_.items():
    structured_feature_names[idx] = name

text_feature_names = [None] * len(text_vectorizer.vocabulary_)
for name, idx in text_vectorizer.vocabulary_.items():
    text_feature_names[idx] = name

label_names = [m.LABEL_TO_PAINTING[i] for i in range(len(m.LABEL_TO_PAINTING))]

arrays = {
    "label_names": np.array(label_names, dtype="<U64"),
    "structured_feature_names": np.array(structured_feature_names, dtype="<U256"),
    "text_feature_names": np.array(text_feature_names, dtype="<U256"),
    "n_estimators": np.array([model.n_estimators], dtype=np.int32),
    "max_depth": np.array([150], dtype=np.int32),
}

for i, est in enumerate(model.estimators_):
    t = est.tree_
    prefix = f"tree_{i:03d}"
    arrays[f"{prefix}_children_left"] = t.children_left.astype(np.int32)
    arrays[f"{prefix}_children_right"] = t.children_right.astype(np.int32)
    arrays[f"{prefix}_feature"] = t.feature.astype(np.int32)
    arrays[f"{prefix}_threshold"] = t.threshold.astype(np.float32)
    arrays[f"{prefix}_value"] = t.value[:, 0, :].astype(np.float32)

np.savez_compressed("predgpt_forest.npz", **arrays)