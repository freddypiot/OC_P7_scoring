
import bentoml
import pandas as pd
import re
import numpy as np
import asyncio
from bentoml.io import Text, JSON  # Correction ici

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

@svc.api(input=int, output=JSON())  # Correction : Utilisation de Text()
async def predict(id_client: str) -> dict:
    """
    Prédit l'acceptation ou le refus de crédit basé sur l'ID client.
    Input attendu : un entier sous forme de texte (ex: "100002") v1
    """
    #try:
    #    id_client = int(id_client)  # Convertir en entier
    #except ValueError:
    #    return {"error": "id_client doit être un entier valide"}

    # Vérifier si le client existe dans les données
    input = df_read[df_read['SK_ID_CURR'] == id_client]
    #if input.empty:
    #    return "Client inconnu"
    #prediction=self.prediction_credit_model.predict(input)[0]
    probabilite = self.prediction_credit_model.predict_proba(input)[0][1]
    prediction = (probabilite >= seuil).astype(int)

    if prediction == 1:
        #reponse = "Client : " + str(id_client) + " --> Crédit refusé"
        reponse = "Crédit refusé"
    else:
        #reponse = "Client : " + str(id_client) + " --> Crédit accordé"
        reponse = "Crédit accordé"

    return {
        "id_client": id_client,
        "refus_credit": prediction,
        "probabilite": round(probabilite, 4)
    }
