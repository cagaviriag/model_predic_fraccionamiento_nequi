# Generación de modelo e inferencia para detectar mala práctica transaccional NEQUI

Este proyecto contiene el desarrollo de un modelo de aprendizaje automático encontrar patrones de un comportamiento inadecuado o Mala Práctica Transaccional, el cual consta fraccionar las transacciones en un número mayor de operaciones con menor monto que agrupadas suman el valor de la transacción original

El ejercicio se desarrolló una parte en spark para tener un conocimiento general del fenómeno, en qué sede o sucursal se presenta más, en que cuentas es más frecuente este comportamiento, en qué mes, etc. 
Adicional se realiza un modelo de ML para identificar y/o predecir cuándo se esté presentando esta mala practica.  


# Tabla de contenido
- [Definición](#definicion)
- [Ampliacion](#ampliacion)
- [Resultados](#resultados)
- [Autor](#autor)

#


## Definición:

    .
    ├── desarrollo-python                  
    │   │── __pycache__           
    │   │── bd                
    │   │── models            
    │   │── s3
    │   │── api.py  
    │   │── api_modelo.py
    │   │── funtions.p 
    │   │── modelado.py 
    │   │── nequi_ds.ipynb  
    │   │── descriptivo.ipynb            
    │   │── requirements.txt      
    │   │── resultado_metricas.xlsx
    │   │── result_predict.xlsx              
    ├── nequi                                    
    ├── Científico de datos en NEQUI                        
    ├── Prueba técnica DS Nequi                                           
    └── .gitignore                 

#


A continuación se detalla el contenido de cada uno de los elementos

## Ampliacion:

###  folders y archivos

*   bd: cuenta con los .parquet compartidos inicialmente y algunos .csv que fueron necesarios exportar .
*   s3: se encuentra las carpetas con los archivos particionados que fuera exportados desde s3 al correr el proceso en aws
*   model: cuenta con los objetos .pkl necesario para la modelación
*   nequi_ds.ipnyb: cuenta con el proceso que se realizó en spark desde una EMR de AWS para procesar los 86 millones de registros
*   descriptivo: tiene toda la información del descriptivo con un fragmento (last 20 días) y la invocación del modelo utilizar con sus respectivos ajustes
*   funtions.py: script con todas las funciones para el tratamiento y visualización utilizadas en el descriptivo.
*   modelo.py: script con toda la modelación realizada.
*   api_modelo.py: main del api que consume el script modelo.py
*   api.py: activado de la api
*   requirements archivo con las librerias y versiones necesarias
#


## Resultados]

Los siguientes son los resultados obtenidos en el benchmark de modelos

Accuracy train	|Accuracy test|	Precision train| Precision test	|Recall train|	Recall test|	F1_score train|	F1_score test|	Roc auc train|	Roc auc test	|modelo|
| ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |
1	|1	|1	|1	|1	|1	|1	|1	|1	|1	|GradientBoostingClassifier|
1	|1	|1	|1	|1	|1	|1	|1	|1	|1	|RandomForestClassifier|
0,999832729	|0,999956635	|0,998285714	|0,999554962	|1	|1	|0,999142122	|0,999777432	|0,999907339	|0,999975978	|LGBMClassifier|
#


## Autor

cris.gaviria@hotmail.com

cagaviriag@gmail.com

* [linkedin](https://www.linkedin.com/feed/)

