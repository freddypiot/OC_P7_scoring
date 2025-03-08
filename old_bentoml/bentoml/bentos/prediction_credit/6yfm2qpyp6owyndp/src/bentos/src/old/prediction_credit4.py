
import pandas as pd
import bentoml
from bentoml.validators import DataframeSchema
from typing_extensions import Annotated
import re
import json


#path = '/content/drive/MyDrive/Colab Notebooks/Formation_DS_OC/P7'
#df_read = pd.read_csv(path + '/export_base_credit.csv', sep="!").head(1000)
df_read = pd.read_csv('export_base_credit_1000.csv', sep="!")
df_read = df_read.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_read = df_read.drop('TARGET', axis=1)
seuil = 0.1

@bentoml.service
class PredictionCredit():
    prediction_credit_model: bentoml.models.get("prediction_credit_model:latest")


    def __init__(self):
        self.prediction_credit_model = bentoml.sklearn.load_model('prediction_credit_model')

    @bentoml.api
    def predict(
        self,
        id_client: int
        ) -> str:
        """
        predict expects id_client (ex : 100002) as input
        """
        input = df_read[df_read['SK_ID_CURR'] == id_client]
        if input.empty:
            return "Client inconnu"
        #prediction=self.prediction_credit_model.predict(input)[0]
        probabilite = self.prediction_credit_model.predict_proba(input)[0][1]
        prediction = (probabilite >= seuil).astype(int)

        if prediction == 1:
            #reponse = "Client : " + str(id_client) + " --> Crédit refusé"
            reponse = "Crédit refusé"
        else:
            #reponse = "Client : " + str(id_client) + " --> Crédit accordé"
            reponse = "Crédit accordé"

        result = {
        "id_client": id_client,
        "refus_credit": int(prediction),
        "probabilite": probabilite.round(4)
        }
        return json.dumps(result)
        #return "Client : " + str(id_client) + " --> " + reponse

# Créer une instance du service
svc = PredictionCredit()
