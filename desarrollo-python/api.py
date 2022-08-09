from fastapi import FastAPI, Query
import uvicorn
from pydantic import BaseModel
import api_modelo


class ParamsInference(BaseModel):
    dictionary: dict = Query(..., description='diccionario con los registros a predecir')


App = FastAPI(
    title="Inferences best model",
    description="Api dirigida a generar prediciones de posibles fraccionamiento",
    version="1.0.1",
)


@App.get("/")
def root():
    return {"Inferences best model 1.0.1"}


@App.post("/inference-model/")
def inference(params: ParamsInference):

    return api_modelo.main_inference(
        dictionary=params.dictionary
    )


if __name__ == "__main__":
    uvicorn.run("api:App", host="192.168.1.78", port=90)