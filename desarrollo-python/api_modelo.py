import pickle
import modelado
import pandas as pd
import pandas as pd
import json
from pandas import json_normalize


def main_inference(dictionary):

    data_predic = pd.DataFrame.from_dict(dictionary, orient ='columns')
    data_predic['transaction_date'] = pd.to_datetime(data_predic['transaction_date'])
    data_predic = data_predic.copy()
    data_predic = data_predic.reset_index(drop=True)
    data_predic = pd.DataFrame(data_predic)


    with open("models/grid_search_rfc.pkl", 'rb') as file:
        grid_search_rfc = pickle.load(file)


    result_predic = modelado.pipeline_prediccion(data_predic=data_predic, obj_model=grid_search_rfc)

    result_predic1 = result_predic[['_id','prediccion','diference_hour']]
    result_predic1.fillna(-999,inplace=True)
    result_predic1 = result_predic1.to_dict()

    return result_predic1

