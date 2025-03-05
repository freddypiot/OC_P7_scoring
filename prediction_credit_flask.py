import pandas as pd
import joblib
import re
import json
from flask import Flask, request, jsonify

# Charger les données
df_read = pd.read_csv("export_base_credit_1000.csv", sep="!")
df_read = df_read.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
df_read = df_read.drop("TARGET", axis=1)
seuil = 0.1  # Seuil de classification

# Charger le modèle
model = joblib.load("saved_model.pkl")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    id_client = data.get("id_client")

    if id_client is None:
        return jsonify({"error": "ID client manquant"}), 400

    input_df = df_read[df_read["SK_ID_CURR"] == id_client]

    if input_df.empty:
        return jsonify({"error": "Client inconnu"}), 404

    probabilities = model.predict_proba(input_df)
    probabilite = probabilities[0][1]
    prediction = int(probabilite >= seuil)

    result = {
        "id_client": id_client,
        "refus_credit": prediction,
        "probabilite": round(probabilite, 4),
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
