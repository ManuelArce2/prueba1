from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
import matplotlib.pyplot as plt
from graphviz import Source
from sklearn.tree import export_graphviz
import os
import io
import base64

app = Flask(__name__)

# Funciones auxiliares
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    return train_set, val_set, test_set


def remove_labels(df, label_name):
    if label_name not in df.columns:
        raise KeyError(f"La columna '{label_name}' no se encuentra en el dataset.")
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return X, y


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/action/<action>", methods=["GET"])
def action(action):
    # Ruta ajustada
    dataset_path = "/home/manuel/Documentos/Programacion_Logia/datasets/TotalFeatures-ISCXFlowMeter.csv"
    if not os.path.exists(dataset_path):
        return jsonify({"error": "El dataset no fue encontrado."}), 404

    df = pd.read_csv(dataset_path)

    if action == "load_data":
        data_head = df.head(10).to_html()
        return jsonify({"message": "Datos cargados", "data": data_head})

    elif action == "length_features":
        data_length = len(df)
        num_features = len(df.columns)
        return jsonify({"message": "Longitud y Características", "length": data_length, "features": num_features})

    elif action == "split_scale":
        try:
            train_set, val_set, test_set = train_val_test_split(df)
            X_train, y_train = remove_labels(train_set, "class")
            X_val, y_val = remove_labels(val_set, "class")
            X_test, y_test = remove_labels(test_set, "class")

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_scaled = DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

            data_scaled_head = X_train_scaled.head(10).to_html()
            return jsonify({"message": "Dataset dividido y escalado", "scaled_data": data_scaled_head})
        except KeyError as e:
            return jsonify({"error": str(e)}), 400

    elif action == "train_tree":
        try:
            train_set, val_set, test_set = train_val_test_split(df)
            X_train, y_train = remove_labels(train_set, "class")
            X_val, y_val = remove_labels(val_set, "class")

            clf_tree = DecisionTreeClassifier(random_state=42)
            clf_tree.fit(X_train, y_train)

            y_train_pred = clf_tree.predict(X_train)
            f1_train = f1_score(y_train, y_train_pred, average="weighted")

            y_val_pred = clf_tree.predict(X_val)
            f1_val = f1_score(y_val, y_val_pred, average="weighted")

            return jsonify({"message": "Modelo entrenado", "f1_train": f1_train, "f1_val": f1_val})
        except KeyError as e:
            return jsonify({"error": str(e)}), 400


@app.route("/train", methods=["POST"])
def train_model():
    dataset_path = "/home/manuel/Documentos/Programacion_Logia/datasets/TotalFeatures-ISCXFlowMeter.csv"
    if not os.path.exists(dataset_path):
        return jsonify({"error": "El dataset no fue encontrado."}), 404

    df = pd.read_csv(dataset_path)
    df["class"], _ = pd.factorize(df["class"])

    train_set, val_set, test_set = train_val_test_split(df)
    X_train, y_train = remove_labels(train_set, "class")
    X_test, y_test = remove_labels(test_set, "class")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    y_pred = rf_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return jsonify({"mse": mse, "r2": r2})


if __name__ == "__main__":
    app.run(debug=True)
