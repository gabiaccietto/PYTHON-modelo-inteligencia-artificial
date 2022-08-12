#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from time import time

#Para entrenamiento en regresion 

from sklearn.svm import SVR

#Metrica de rendimiento
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split


# In[23]:


df=pd.read_csv(r'C:\Users\Usuario\Downloads\datos_limpios.csv')
df.head()


# In[24]:


list_x=['gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'SeniorCitizen'
       
       ]


# In[25]:


X= df[list_x]
X.head()


# In[26]:


list_y=['MonthlyCharges']


# In[27]:


y=df[list_y]
y.head()


#                                                  O PUEDO HACER ESTO :

# In[28]:


X=df.iloc[:,0:len(df.columns)-1]
y=df[['MonthlyCharges']]


# In[29]:


X.head()


# In[30]:


y.head()


# In[31]:


#separamos los datros de entrenamiento y prueba 

X_train,X_test,y_train,y_test= train_test_split(X,y)


# In[64]:


#definimos el algoritmo de regresion

regresor=SVR(kernel='rbf') #probar con linear, poly o rbf
hora_inicio=time()

#entrenamiento del algoritmo
regresor.fit(X_train.values, y_train.values.ravel())

print(f'El entrenamiento tardo {time() - hora_inicio} segundos' )


# In[65]:


X_train.info #cuantos dstos uso para entrenar 


# In[66]:


y_pred = regresor.predict(X_test)


# In[67]:


probar=1000

X_axis= np.arange(probar)
fig, ax= plt.subplots()
ax.scatter(X_axis,y_test.iloc[0:probar].values)
ax.scatter(X_axis,y_pred[0:probar])
plt.show()


# In[68]:


r2_score(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




