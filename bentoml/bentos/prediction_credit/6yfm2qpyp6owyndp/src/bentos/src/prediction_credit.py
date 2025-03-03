
import pandas as pd
import bentoml
import re
import json

# Charger les données
df_read = pd.read_csv('export_base_credit_1000.csv', sep="!")
df_read = df_read.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
df_read = df_read.drop('TARGET', axis=1)
seuil = 0.1  # Seuil de classification

@bentoml.service
class PredictionCredit():
    def __init__(self):
        # Charger le modèle directement dans self
        self.prediction_credit_model = bentoml.sklearn.load_model("prediction_credit_model:latest")

    @bentoml.api
    def predict(self, id_client: int) -> str:
        """
        Prédit l'acceptation ou le refus de crédit basé sur l'ID client.
        Input attendu : {"id_client": 100002}
        """
        input_df = df_read[df_read['SK_ID_CURR'] == id_client]

        if input_df.empty:
            return json.dumps({"error": "Client inconnu"})

        # Utiliser directement `self.prediction_credit_model` au lieu du Runner
        probabilities = self.prediction_credit_model.predict_proba(input_df)
        probabilite = probabilities[0][1]
        prediction = int(probabilite >= seuil)

        result = {
            "id_client": id_client,
            "refus_credit": prediction,
            "probabilite": round(probabilite, 4)
        }
        return json.dumps(result)

# Créer une instance du service
svc = PredictionCredit()
