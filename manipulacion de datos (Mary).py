import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


tabla_nombres=pd.read_csv(r'C:\Users\Usuario\Downloads\nombres.csv')
tabla_nombres.head()






#saber cuantos hombres tienen el nombre de mary
tabla_mary=tabla_nombres[
    (tabla_nombres['sex']== 'M') &
    (tabla_nombres ['name']== 'Mary')
    
    
    ]
tabla_mary.head()

#peir datos que me interesan de columnas 
data_interest=tabla_mary[['state','quantity']]
data_interest.head()










#-------------------------------------------------







data_interest[['quantity']].sum()


estados=data_interest.groupby('state').agg('sum')
estados.head()