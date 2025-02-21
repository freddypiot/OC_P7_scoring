import pandas as pd
import bentoml
from bentoml.io import JSON
from fastapi import FastAPI
import re
import json

# Charger les données
df_read = pd.read_csv('export_base_credit_1000.csv', sep="!")
df_read = df_read.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
df_read = df_read.drop('TARGET', axis=1)
seuil = 0.1

# Créer une instance de FastAPI
app = FastAPI()

# Charger le modèle
model = bentoml.sklearn.load_model('prediction_credit_model')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(id_client: int):
    input = df_read[df_read['SK_ID_CURR'] == id_client]
    if input.empty:
        return {"message": "Client inconnu"}
    
    probabilite = model.predict_proba(input)[0][1]
    prediction = (probabilite >= seuil).astype(int)

    result = {
        "id_client": id_client,
        "refus_credit": int(prediction),
        "probabilite": probabilite.round(4)
    }
    return result

# Créer un service BentoML
svc = bentoml.Service("prediction_credit_service", runners=[])

# Monter l'application FastAPI
svc.mount_asgi_app(app)