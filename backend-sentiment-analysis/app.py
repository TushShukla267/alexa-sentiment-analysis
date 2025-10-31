import os
import io
import traceback
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np

# --- Config ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # if backend/ used; adjust if app.py at repo root
# If you put app.py at repo root, change BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Files expected relative to BASE_DIR:
VECTORIZER_PATH = os.path.join(BASE_DIR, "countVectorizer.pkl")
MODEL_XGB_PATH = os.path.join(BASE_DIR, "model_xgb.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

ALLOWED_TEXT_EXTS = {".csv"}

app = Flask(__name__)
CORS(app)  # opens to all origins — adjust in production

# --- Helpers to load artifacts safely ---
def safe_load(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            # try with pickle as fallback
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

vectorizer = safe_load(VECTORIZER_PATH)
model_xgb = safe_load(MODEL_XGB_PATH)
scaler = safe_load(SCALER_PATH)  # optional, used if model expects scaled features

# Map of available models for 'model' param. Add more if you save them.
MODEL_MAP = {
    "xgb": model_xgb
    # "rf": model_rf, "dt": model_dt  # if you create and save these, add here
}

def pick_model(model_key: str):
    m = MODEL_MAP.get(model_key)
    if m is None:
        # fallback to whatever we loaded (xgb)
        return model_xgb
    return m

def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic to find a likely text column in incoming CSV:
    - prefer columns containing 'text' or 'review' (case-insensitive)
    - otherwise pick the first object/string dtype column
    """
    cols = df.columns.tolist()
    # check names
    for col in cols:
        if any(x in col.lower() for x in ("text", "review", "comment", "content")):
            return col
    # fallback: first object column
    for col in cols:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            return col
    return None

def text_to_features(texts):
    """
    Convert list/Series of texts into model input using loaded vectorizer (CountVectorizer or similar).
    Returns a sparse/dense matrix compatible with model.predict.
    """
    if vectorizer is None:
        raise RuntimeError("Vectorizer is not loaded on server.")
    X = vectorizer.transform(texts)
    # if scaler is present and expects dense arrays (rare for pure text pipelines),
    # attempt to convert to array and scale
    if scaler is not None:
        try:
            X = X.toarray() if hasattr(X, "toarray") else np.array(X)
            X = scaler.transform(X)
        except Exception:
            # ignore scaler errors for text-only models
            pass
    return X

def predict_single_text(model, text: str):
    X = text_to_features([text])
    # predict probability if possible
    prob = None
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # take positive class probability; infer which column is positive
            # If two columns, assume second corresponds to positive class label '1'
            if proba.shape[1] == 2:
                prob = float(proba[0, 1])
                probs = {"0": float(proba[0, 0]), "1": float(proba[0, 1])}
            else:
                # multiclass fallback: provide highest class probability
                idx = int(np.argmax(proba[0]))
                prob = float(proba[0, idx])
                probs = {str(i): float(p) for i, p in enumerate(proba[0].tolist())}
        else:
            prob = None
    except Exception:
        # some models (raw xgboost booster saved differently) need special handling
        try:
            proba = model.predict(X)
            prob = float(proba[0])
        except Exception:
            prob = None

    # get prediction label
    try:
        pred = model.predict(X)
        # if pred is array of classes, take first
        label = pred[0]
    except Exception:
        label = None

    return label, prob, probs

# --- Routes ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts multipart/form-data:
    - text: string (single prediction)
    - file: CSV upload (predict for rows)
    - model: optional string key (xgb/rf/dt)
    """
    try:
        chosen_model_key = (request.form.get("model") or "xgb").lower()
        model = pick_model(chosen_model_key)
        if model is None:
            return jsonify({"error": "No model loaded on server."}), 500

        # Prefer file if present (CSV)
        if "file" in request.files and request.files["file"].filename:
            f = request.files["file"]
            filename = f.filename
            _, ext = os.path.splitext(filename)
            if ext.lower() not in ALLOWED_TEXT_EXTS:
                return jsonify({"error": "Only CSV files are supported."}), 400

            # read CSV into pandas
            try:
                # support variable encodings and separators
                content = f.read()
                # try utf-8 then fallback
                try:
                    s = content.decode("utf-8")
                except Exception:
                    s = content.decode("latin1")
                # read with pandas: try common separators
                df = None
                for sep in [",", ";", "\t"]:
                    try:
                        df = pd.read_csv(io.StringIO(s), sep=sep, engine="python")
                        if df is not None and df.shape[1] > 0:
                            break
                    except Exception:
                        df = None
                if df is None:
                    return jsonify({"error": "Could not parse uploaded CSV."}), 400
            except Exception as e:
                return jsonify({"error": "Failed to read uploaded file.", "detail": str(e)}), 400

            text_col = detect_text_column(df)
            if text_col is None:
                return jsonify({"error": "Could not detect text column in CSV."}), 400

            texts = df[text_col].fillna("").astype(str).tolist()
            X = text_to_features(texts)
            out_preds = []
            # compute probabilities if possible in batch
            probs_batch = None
            try:
                if hasattr(model, "predict_proba"):
                    probs_batch = model.predict_proba(X)
                preds = model.predict(X)
            except Exception:
                # try using model.predict on sparse/dense mismatch
                preds = model.predict(X)
            # build output list
            for i, t in enumerate(texts):
                p = preds[i]
                prob = None
                probs = None
                if probs_batch is not None:
                    row = probs_batch[i]
                    if row.shape and len(row) == 2:
                        prob = float(row[1])
                        probs = {"0": float(row[0]), "1": float(row[1])}
                    else:
                        # multiclass
                        probs = {str(idx): float(val) for idx, val in enumerate(row.tolist())}
                        prob = max(probs.values()) if probs else None
                out_preds.append({
                    "text": t,
                    "prediction": int(p) if (isinstance(p, (np.integer, int))) else str(p),
                    "probability": prob,
                    "probs": probs
                })
            return jsonify({"model": chosen_model_key, "predictions": out_preds}), 200

        # Else if single text provided
        text = (request.form.get("text") or "").strip()
        if text:
            label, prob, probs = predict_single_text(model, text)
            # normalize label: if numeric 1/0 return int, else leave string
            out_label = None
            if isinstance(label, (np.integer, int)):
                out_label = int(label)
            elif isinstance(label, (np.ndarray, list)):
                out_label = label[0]
            else:
                out_label = label

            # also if model used label mapping maybe 1 means positive
            return jsonify({
                "prediction": out_label,
                "probability": None if prob is None else float(prob),
                "probs": probs,
                "model": chosen_model_key
            }), 200

        return jsonify({"error": "No text or file provided."}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Server error", "detail": str(e)}), 500


if __name__ == "__main__":
    # When running development: flask app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
