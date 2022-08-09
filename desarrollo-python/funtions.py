import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#############################################################################
############################### Lectura dataframe ###########################
#############################################################################
#data = pd.read_csv(r"bd\last_20_days.csv")
#data['transaction_date'] = pd.to_datetime(data['transaction_date'])
#data.shape
#data.columns

#data_predic = pd.read_csv(r"bd\df_test.csv")
#data_predic['transaction_date'] = pd.to_datetime(data_predic['transaction_date'])
#data_predic.shape


#############################################################################
########################### tratamiento diferencias ########################
#############################################################################
def diference_time(df_model):
    
    df_model['transaction_date'] = pd.to_datetime(df_model['transaction_date'])
    df_model.sort_values(by=['transaction_date'],inplace=True,ascending=True)

    df_model['subsidiary_account_number'] = df_model['subsidiary'] + "-" + df_model['account_number']
    df_model['diference_hour'] = df_model.groupby(['subsidiary', 'account_number'])['transaction_date'].diff().dt.seconds.div(3600).reset_index(drop=True)
    df_model['targer'] = np.where(df_model['diference_hour']< 12 , "Fraude","No fraude")

    return df_model




#############################################################################
########################### descripcion de los datos ########################
#############################################################################
def tabla_resumen(df):
    """Pinta resumenes basicos del df como el tamaño, duplicado, faltantes etc.

    Args:
        df (df): datafra para calcular unos resumen basicos
    """

    print("-------------------------------------------------------------------------")
    print("Dimension base de datos                    {}".format(df.shape[0]))
    print("Cantida de comercios unicas                {}".format(df['merchant_id'].value_counts().count())) ## Comercio
    print("Cantida de sucursales unicas               {}".format(df['subsidiary'].value_counts().count())) ## sucursal
    print("fecha minina                               {}".format(df['transaction_date'].min())) ## fechas 
    print("fecha maxima                               {}".format(df['transaction_date'].max())) ## fechas
    print("Cuentas destino unicas                     {}".format(df['account_number'].value_counts().count())) ## cuenta destino
    print("Id usuaruias cuentas destino unicas        {}".format(df['user_id'].value_counts().count())) ## usuario
    print("Tipo transaacion                           {}".format(df['transaction_type'].value_counts().count())) ## tipo 
    print("-------------------------------------------------------------------------")

    return 



#############################################################################
################# Grafica categorica para una variable ######################
#############################################################################
def grafico_cat_univariado(df,x):
    """Pinta el grafico por la varible desea y una tabla resumen

    Args:
        df (df): datafra a trabajar 
        x (str): str con el nombre de la variables a pintar

    Returns:
        _type_: grafico
    """
    print("Mala Practica Transaccional:\n", df[x].value_counts(normalize=True), sep = "")
    plt.figure(figsize = (10,10))
    sns.countplot(x = x, data = df, order = df[x].value_counts(normalize=True).index)
    plt.xticks(rotation = 45)
    plt.xlabel("Resultado", fontsize = 12)
    plt.ylabel("Cantidad ", fontsize = 12)# fontweight = "black"
    plt.title("Mala Practica Transaccional", fontsize=18, color="black")
    return plt.show()



#############################################################################
#### Grafica categorica para dos variables vs (targer) ##########
#############################################################################
def grafico_cat_dos_variables(df,x,y,only_table):
    """Pinta el grafico por las varibles deseadas agrupando x; transponiendo y


    Args:
        df (df): datafra a trabajar 
        x (str): str con variable a agrupar 
        y (str): str con variable a segmentar y posteriomente a transponer 

    Returns:
        _type_: _grafico y tabla resumen_
    """
    

    plt.figure(figsize = (15,15))
    df1 = df.groupby(x)[y].value_counts(normalize=True)
    df1 = df1.rename('percent').reset_index()
    g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1,height=7).set(title='\n Resultado por {}'.format(x))
    g.ax.set_ylim(0,1)
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=90)

    if only_table:
            pivoted = df1.pivot(index=x, columns=y)
            print(pivoted)
            return 

    else:
        for p in g.ax.patches:
            txt = str(p.get_height().round(2)) + '%'
            txt_x = p.get_x()
            txt_y = p.get_height()
            g.ax.text(txt_x,txt_y,txt)
            pivoted = df1.pivot(index=x, columns=y)
            print(pivoted)
            return plt.show()
    
    return



#############################################################################
#### Grafica categorica para dos variables vs (targer) ##########
#############################################################################
def grafico_cat_dos_variables_table(df,x,y):
    """Pinta el grafico por las varibles deseadas agrupando x; transponiendo y


    Args:
        df (df): datafra a trabajar 
        x (str): str con variable a agrupar 
        y (str): str con variable a segmentar y posteriomente a transponer 

    Returns:
        _type_: _grafico y tabla resumen_
    """

    df1 = df.groupby(x)[y].value_counts(normalize=True)
    df1 = df1.rename('percent').reset_index()
    pivoted = df1.pivot(index=x, columns=y)
             
    return pivoted



#############################################################################
######### Grafica boxplot variable numerica vs targer ###########
#############################################################################
def grafico_box_plot(df,lista=list()):
    """Grafico de box plot multiples variables vs 'targer'

    Args:
        df (df): datafra a trabajar
        lista (_type_, optional): _Lista con los nombres de las variables a graficar_. Defaults to list().

    Returns:
        _type_: _grafico de box plot con las variables deseadas_
    """

    plt.figure(figsize=(15, 15))
    plt.suptitle("Boxplots de las variables: transaction_amount y diference_hour ",
                 fontsize=18)
    for cant, posicion in enumerate(lista):

        plt.subplot(1, 2, cant+1)
        sns.boxplot(data = df, x = 'targer', y = str(lista[cant]))
        plt.xlabel('Resultado')
        plt.ylabel(lista[cant])
    return plt.show()



#############################################################################
######### Tabla resumen variable numerica vs targer #############
#############################################################################
def tabla_resumen_numericas(df,lista=list()):
    """tabla resumen de variables vs 'targer'

    Args:
        df (df): datafra a trabajar
        lista (_type_, optional): _Lista con los nombres de las variables a resumir_. Defaults to list().

    Returns:
        _type_: _df con los resultados_
    """
    result = pd.DataFrame()
    for cant in np.arange(0,len(lista)):

        tabla = pd.DataFrame(df.groupby('targer')[lista[cant]].describe())
        tabla = tabla.drop(columns='count')
        tabla['coef_variacion'] = tabla['std'] / tabla['mean']
        tabla['IQR'] = tabla['75%'] - tabla['25%']
        tabla['Ati_sup'] = tabla['75%'] + 1.5 * tabla['IQR']
        tabla['metrica'] = lista[cant]
        result = pd.concat([result,tabla])
    return result



#############################################################################
####################### Analisis de correlacion #############################
#############################################################################
def analisis_correlacion(bd_numerica):
    """_Calcula y pinta la correlacion de las variables ingresadas_

    Args:
        bd_numerica (df): _df con variables solo numericas_

    Returns:
        _type_: _gradico de correlacion_
    """
    corr = bd_numerica.corr('spearman')
    plt.figure(figsize=(15,15))
    sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8})
    return plt.show()



#############################################################################
############################### Plot density ################################
#############################################################################
def plot_density(df,var):
    """_Grafica la densidad de la variable deseada_

    Args:
        df (df): datafra a trabajar
        var (str): str con el nombre de la variable que desea graficar

    Returns:
        _type_: _grafico de densidad_
    """
    df=df
    sns.displot(df, x=var, kind="kde")
    plt.title("Density {}".format(var))
    plt.xlabel('Resultado')
    plt.ylabel(var)

    return plt.show()


