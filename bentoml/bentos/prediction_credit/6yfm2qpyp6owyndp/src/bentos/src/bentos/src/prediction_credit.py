
import bentoml
import pandas as pd
import json
import re
from bentoml.io import JSON
from sklearn.base import BaseEstimator

# Charger les données
df_read = pd.read_csv("export_base_credit_1000.csv", sep="!")
df_read = df_read.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '', x))
df_read = df_read.drop("TARGET", axis=1)
seuil = 0.1  # Seuil de classification

# Charger le modèle BentoML
model_ref = bentoml.sklearn.get("prediction_credit_model:latest")
model_runner = model_ref.to_runner()

# Définir le service BentoML
svc = bentoml.Service("prediction_credit_service", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def predict(input_data: dict) -> dict:
    """
    Prédit l'acceptation ou le refus de crédit basé sur l'ID client.
    Input attendu : {"id_client": 100002}
    """
    id_client = input_data.get("id_client")

    input_df = df_read[df_read["SK_ID_CURR"] == id_client]
    if input_df.empty:
        return {"error": "Client inconnu"}

    # Prédiction
    probabilities = model_runner.predict_proba.run(input_df)
    probabilite = probabilities[0][1]
    prediction = int(probabilite >= seuil)

    return {
        "id_client": id_client,
        "refus_credit": prediction,
        "probabilite": round(probabilite, 4)
    }
