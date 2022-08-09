from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import datetime
import matplotlib.pyplot as plt
import category_encoders as ce


#############################################################################
############################### Lectura dataframe ###########################
#############################################################################
#df = pd.read_csv(r"C:\Users\user\Desktop\prueba-cristian-gaviria\desarrollo-python\bd\last_20_days.csv")
#df['transaction_date'] = pd.to_datetime(df['transaction_date'])

#data_predic = pd.read_csv(r"bd\df_test.csv")
#data_predic['transaction_date'] = pd.to_datetime(data_predic['transaction_date'])


#############################################################################
########################## eliminacion_atipicos #############################
#############################################################################
def eliminacion_atipicos(df,var):
    """_eliminacion de atipcos IQR = (quantile(0.9) - quantile(0.1))
    minimo: quantile(0.1) - (1.5 * IQR)
    maximo: quantile(0.9) + (1.5 * IQR) 
    teniendo en cuenta los limites anteriores, esta funcion tiene quedas las variables
    por las que va a trabajar.
    _

    Args:
        df (df): dataframe a trabajar

    Returns:
        _type_: _returna df sin los datos atipicos_
    """
    data_seleccion = df[var]
    data_no_seleccion = df.drop(columns=var)
    mínimo = data_seleccion.quantile(0.1)
    máximo = data_seleccion.quantile(0.9)
    IQR = máximo - mínimo
    clean_select = data_seleccion[~ ((data_seleccion < (mínimo - 1.5 * IQR)) | (data_seleccion > (máximo + 1.5 * IQR))).any(axis=1)].copy()
    df_model = clean_select.merge(data_no_seleccion,left_index=True, right_index=True,how = 'left')
    return df_model





entrenamiento = False
if entrenamiento:

    df_model = eliminacion_atipicos(df=df,var=['transaction_amount'])
    df_model = df
    df_model.shape
    #############################################################################
    ############################## Modelacion ###################################
    #############################################################################
    df_model.sort_values(by=['transaction_date'],inplace=True,ascending=True)
  
    df_model['diference_hour'] = df_model.groupby(['subsidiary', 'account_number'])['transaction_date'].diff().dt.seconds.div(3600).reset_index(drop=True)
    df_model['targer'] = np.where(df_model['diference_hour']< 12 , "1","0")
    
    
    ###########
    ## TODO: pilas
    X = df_model.drop(columns=['targer'])
    y = df_model['targer']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42, stratify=y)
    X_train['targer'] = y_train
    
    df_model = X_train
    df_model.shape

    bd_no_process = df_model[['targer','diference_hour','transaction_amount']]
    bd_scales = df_model[['transaction_amount']]


    #############################################################################
    ############################# StandardScaler ################################
    #############################################################################
    scaler = StandardScaler()
    scaler.fit(bd_scales)
    df_scales_frame = pd.DataFrame(scaler.transform(bd_scales),columns=bd_scales.columns)

    with open("models/scaler_model.pkl", 'wb') as file:
        pickle.dump(scaler, file)

    with open("models/scaler_model.pkl", 'rb') as file:
        scaler = pickle.load(file)


    #############################################################################
    ############################# TargetEncoder ################################
    #############################################################################

    #bd_TargetEncoder.rename(columns={'targer':'resultado'},inplace=True)
    #bd_TargetEncoder['resultado'] = "targe_" + bd_TargetEncoder['resultado']

    #target_encoder = ce.TargetEncoder(cols=['transaction_type'],smoothing=0, return_df=True)
    #target_encoder.fit(bd_TargetEncoder['transaction_type'], bd_TargetEncoder['resultado'])
    #df_TargetEncoder_frame = target_encoder.transform(bd_TargetEncoder['transaction_type'], bd_TargetEncoder['resultado'])
    
    #with open("models/target_encoder_model.pkl", 'wb') as file:
    #    pickle.dump(target_encoder, file)

    #with open("models/target_encoder_model.pkl", 'rb') as file:
    #    target_encoder = pickle.load(file)


    #############################################################################
    ############################# Consolodacion BD ###############################
    #############################################################################
    df_final_model = pd.concat([bd_no_process, df_scales_frame], axis=1, join='inner')
    df_final_model = bd_no_process
    df_final_model.fillna(999,inplace=True)


    X = df_final_model.drop(columns=['targer'])
    y = df_final_model['targer']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)


    #smote = SMOTE(random_state = 11)
    #X_train, y_train = smote.fit_resample(X_train, y_train)

    with open("models/X_test.pkl", 'wb') as file:
        pickle.dump(X_test, file)

    with open("models/y_test.pkl", 'wb') as file:
        pickle.dump(y_test, file)


    #################################################################
    ############################ predicciones #######################
    #################################################################
    def metricas_validacion(x_test_pred,y_test_pred,y_train,y_test,modelo):

        Accuracy_train = pd.DataFrame.from_dict({'Accuracy train': [metrics.accuracy_score(y_train, x_test_pred)]})
        Accuracy_test = pd.DataFrame.from_dict({'Accuracy test': [metrics.accuracy_score(y_test, y_test_pred)]})

        Presicion_train = pd.DataFrame.from_dict({'Precision train': [metrics.precision_score(y_train, x_test_pred, pos_label='1')]})
        Presicion_test = pd.DataFrame.from_dict({'Precision test': [metrics.precision_score(y_test, y_test_pred, pos_label='1')]})
        #el total de las prediciones positivas cuantas fueron positivas

        Recall_train = pd.DataFrame.from_dict({'Recall train': [metrics.recall_score(y_train, x_test_pred, pos_label='1')]})
        Recall_test = pd.DataFrame.from_dict({'Recall test': [metrics.recall_score(y_test, y_test_pred, pos_label='1')]})
        ## del total de los casos reales cuantos capturo el modelo

        F1_score_train = pd.DataFrame.from_dict({'F1_score train': [metrics.f1_score(y_train, x_test_pred, pos_label='1')]})
        F1_score_test = pd.DataFrame.from_dict({'F1_score test': [metrics.f1_score(y_test, y_test_pred, pos_label='1')]})

        Roc_auc_train = pd.DataFrame.from_dict({'Roc auc train': [metrics.roc_auc_score(y_train, x_test_pred)]})
        Roc_auc_test = pd.DataFrame.from_dict({'Roc auc test': [metrics.roc_auc_score(y_test, y_test_pred)]})

        resultado = pd.concat([Accuracy_train,Accuracy_test,Presicion_train,Presicion_test,
                                Recall_train,Recall_test,F1_score_train,F1_score_test,
                                Roc_auc_train,Roc_auc_test],axis=1)

        resultado['modelo'] = modelo

        return resultado


    #################################################################
    #########################  LGBMClassifier #######################
    #################################################################
    param_grid = {
        'n_estimators': [300,400],
        'colsample_bytree': [0.7, 0.8],
        'max_depth': [10,15],
        'num_leaves': [40,50],
        'reg_alpha': [1, 1.1],
        'reg_lambda': [1, 1.1],
        'min_split_gain': [0.2,0.3],
        'subsample': [0.6, 0.7],
        'subsample_freq': [20]
    }


    model_lgb = lgb.LGBMClassifier()
    grid_search_lgb = GridSearchCV(estimator=model_lgb, param_grid=param_grid,cv=4, n_jobs=-1, verbose=2)
    grid_search_lgb.fit(X_train,y_train)

    with open("models/grid_search_lgb.pkl", 'wb') as file:
        pickle.dump(grid_search_lgb, file)

    with open("models/grid_search_lgb.pkl", 'rb') as file:
        grid_search_lgb = pickle.load(file)

    grid_search_lgb.best_estimator_
    x_test_pred = pd.DataFrame(grid_search_lgb.predict(X_train))
    y_test_pred = pd.DataFrame(grid_search_lgb.predict(X_test))

    resultado_lgb = metricas_validacion(x_test_pred=x_test_pred,y_test_pred=y_test_pred,
                                       y_train=y_train,y_test=y_test,
                                       modelo='LGBMClassifier')




    #################################################################
    ################# RandomForestClassifier ########################
    #################################################################
    param_grid = {
        'bootstrap': [True],
        'max_depth': [70,80, 90],
        'max_features': [2, 3],
        'min_samples_leaf': [2, 3],
        'min_samples_split': [6, 8, 10, 12],
        'n_estimators': [100,200, 300]
    }

    rf = RandomForestClassifier()
    grid_search_rfc = GridSearchCV(estimator=rf, param_grid=param_grid,cv=4, n_jobs=-1, verbose=2)
    grid_search_rfc.fit(X_train,y_train)
    
    
    grid_search_rfc.best_estimator_
    valores = grid_search_rfc.best_estimator_.feature_importances_
    importancia = pd.DataFrame(valores)

    importancia.index = X_train.columns
    importancia
    
    with open("models/grid_search_rfc.pkl", 'wb') as file:
        pickle.dump(grid_search_rfc, file)

    with open("models/grid_search_rfc.pkl", 'rb') as file:
        grid_search_rfc = pickle.load(file)


    x_test_pred = pd.DataFrame(grid_search_rfc.predict(X_train))
    y_test_pred = pd.DataFrame(grid_search_rfc.predict(X_test))

    resultado_fr = metricas_validacion(x_test_pred=x_test_pred,y_test_pred=y_test_pred,
                                       y_train=y_train,y_test=y_test,
                                       modelo='RandomForestClassifier')


    #################################################################
    #################  GradientBoostingClassifier ###################
    #################################################################
    param_grid = {
        'n_estimators': [150,200],
        'max_depth': [10,15],
        'max_leaf_nodes': [35,50]
    }

    model_xgb = GradientBoostingClassifier()
    grid_search_xgb = GridSearchCV(estimator=model_xgb, param_grid=param_grid,cv=4, n_jobs=-1, verbose=2)
    grid_search_xgb.fit(X_train,y_train)
    

    with open("models/grid_search_xgb.pkl", 'wb') as file:
        pickle.dump(grid_search_xgb, file)

    with open("models/grid_search_xgb.pkl", 'rb') as file:
        grid_search_xgb = pickle.load(file)

    grid_search_xgb.best_estimator_
    x_test_pred = pd.DataFrame(grid_search_xgb.predict(X_train))
    y_test_pred = pd.DataFrame(grid_search_xgb.predict(X_test))
    y_test_pred.value_counts()

    resultado_xgb = metricas_validacion(x_test_pred=x_test_pred,y_test_pred=y_test_pred,
                                       y_train=y_train,y_test=y_test,
                                       modelo='GradientBoostingClassifier')


    
    resultado_final = pd.concat([resultado_xgb,resultado_fr,resultado_lgb])
    resultado_final.to_excel('resultado_metricas.xlsx',index=False)





##################################################################
##################### funcion predicciones  ######################
##################################################################
def pipeline_prediccion(data_predic, obj_model):
    """Pipeline basica para gerar el preprocesing necesario a la df test y posterior predicion con el object tipo modelo
    que le ingresa como parametro

    Args:
        data_predic (df): df a realizar el preprocesamiento
        obj_ohe (_class_): _class con los elementos necesarios para realizar el ohe_
        obj_scale (_class_): _class con los elementos necesarios para realizar el scale_
        obj_model (_class_): _class con el elmento modelo para realizar las prediciones_

    Returns:
        _type_: _retorna el df orginal con el resultado de la prediccion_
    """
    data_predic['transaction_date'] = pd.to_datetime(data_predic['transaction_date'])
    data_predic['status'] = 'new'
    
    last_25_hour = pd.read_csv(r"C:\Users\user\Desktop\prueba-cristian-gaviria\desarrollo-python\bd\last_25_hours.csv")
    last_25_hour['transaction_date'] = pd.to_datetime(last_25_hour['transaction_date'])
    last_25_hour['status'] = 'old'

    df_model = pd.concat([last_25_hour,data_predic])
    df_model.sort_values(by=['transaction_date'],inplace=True,ascending=True)
    df_model = df_model.reset_index(drop=True)

    df_model['subsidiary'] = df_model['subsidiary'].astype(str)
    df_model['subsidiary'] = df_model['subsidiary'].str.strip()

    df_model['account_number'] = df_model['account_number'].astype(str)
    df_model['account_number'] = df_model['account_number'].str.strip()

    df_model['diference_hour'] = df_model.groupby(['subsidiary', 'account_number'])['transaction_date'].diff().dt.seconds.div(3600).reset_index(drop=True)
    df_model =  df_model[df_model['status'] == 'new']
    df_model = df_model.reset_index(drop=True)

    bd_no_process_predic = df_model[['diference_hour','transaction_amount']]

    df_final_predi = bd_no_process_predic[['diference_hour','transaction_amount']]
    df_final_predi.fillna(999,inplace=True)
 
    pred = pd.DataFrame(obj_model.predict(df_final_predi))
    pred = pred.rename(columns={0:'prediccion'})
    df_model['prediccion'] = pred

    return df_model




##################################################################
##################### Grafica Curva ROC  #########################
##################################################################
def grafica_curva_auc(clases_reales, predicciones, predicciones_probabilidades):
    """_Pinta la curva ROC data las prediciones de un modelo vs su test_

    Args:
        clases_reales (df): y reales
        predicciones (df): y predichas
        predicciones_probabilidades (array): probabilidades de pertencer a una clase o no dada en la prediccion  

    Returns:
        _type_: _retorna grafica con la curva ROC_
    """
    fpr, tpr, _ = metrics.roc_curve(clases_reales, predicciones_probabilidades[:, 1], pos_label='1')
    roc_auc = metrics.roc_auc_score(clases_reales, predicciones)
    plt.figure()

    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="estimador aleatorio")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR (recall)')
    plt.title('Curva ROC Best Model Random Forest')
    plt.legend(loc="lower right")
    return plt.show()



