#!/usr/bin/env python
# coding: utf-8

# In[1]:



#MANIPULACION DE DATOS 
import pandas as pd

#OPERACIONES NUMERICAS 
import numpy as np

#CREACION DE GRAFICOS
import matplotlib.pyplot as plt
from time import time

#HERRAMIENTAS SVM
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix


# In[2]:



df= pd.read_csv(r'C:\Users\Usuario\Downloads\digitos.csv')


# In[4]:


df.info


# In[3]:


df.head()


# In[4]:


df['label'].value_counts ()


# In[5]:


df['label'].hist()


# In[6]:


#Obtener una seccion del DF usando numeros de la columna 

pixeles= df.iloc[:,1:785] #x
digitos=df.iloc[:,0:1] #y


# In[7]:


#Separar datos de entrenamiento y prueba 

X_train, X_test, y_train, y_test= train_test_split(
    pixeles,
    digitos,
    test_size=0.5)


# In[8]:


#Consultar info de la muestra de entrenamiento

X_train.shape


# In[9]:


X_train.head()


# In[10]:


def mostrar_num(in_data):
    '''genera un grafico que muestra un registro del set e datos..
    
        Para ello, convierte el array de 1 dimension en una matriz de 28x28
    
    '''
    matriz=np.array(in_data.values)
    plt.imshow(matriz.reshape(28,28))


# In[11]:


mostrar_num(X_test.iloc[100])


# In[12]:


#Creacion del modelo 
modelo=SVC(kernel='linear')


# In[13]:


#Entrenamiento 
hora_inicio=time()
modelo.fit(X_train.values, y_train.values.ravel())
print('entrenamiento terminado en {}'.format(time()- hora_inicio))


# In[14]:


#Crear prediccion de datos

hora_inicio= time()
y_pred=modelo.predict(X_test.values)
print('Prediccionterminada en {}'.format(time()-hora_inicio))
print (y_pred.shape)


# In[15]:


precision=accuracy_score(y_test,y_pred)
print(f'Precision: {precision}')


# In[19]:


prueba=X_test.iloc[135]

print(f'El resultado era: {y_test.iloc[135]}')

mostrar_num(prueba)
prediccion=modelo.predict([prueba])
print(f'El digito es: {prediccion}')


# In[20]:


#Prueba con la matriz de confusion 

conf=confusion_matrix(y_test,y_pred)


# In[21]:


conf


# In[27]:


def plot_cm(cm,classes):
    """Esta funcion se encarga de generar un grafico con nuestra ,atriz de confusion..
    
    cm es la matriz generada por confusion_matrix 
    classes es una lista que contiene las posibles clases que puede predecir nuestro modelo 
    """
    
    plt.imshow(cm,cmap=plt.cm.Blues)
    plt.title("Matriz de confusion")
    plt.colorbar()
    tick_marks= np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks,classes)
    thresh=cm.max()/2.
    for indice_fila,fila in enumerate(cm):
        for indice_columna, columna in enumerate (fila):
            if cm[indice_fila,indice_columna] > thresh:
                color ="white"
            else:
                color="black"
            plt.text(
                indice_columna,
                indice_fila,
                cm[indice_fila, indice_columna],
                color=color,
                horizontalalignment="center"
                
            )
        
    plt.ylabel("valores reales")
    plt.xlabel("valores calculados")
    plt.show()


# In[28]:


plot_cm(conf,[0,1,2,3,4,5,6,7,8,9])


# In[ ]:




