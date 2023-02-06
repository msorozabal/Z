# Descomentar estas 3 lineas si es la primera vez que se ejecuta. 

#!pip install tensorflow
#!pip install tensorflow_addons
#!pip install pydot

import datetime
import os
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)

import seaborn as sns
from sklearn import metrics
 
df = pd.read_csv("backend-dev-data-dataset.txt", sep=",", low_memory=False)

## Reemplazamos los valores nulos con la mediana de la columna antes de partir el dataset

df.replace('na', None, inplace=True)

median = df.median()
df.fillna(median, inplace=True)

train, test = train_test_split(df, test_size=0.1, random_state=24)

#Selecciono mi variable target {cat_8} y particiono los datos crudos en Train, Test

X_train = train.iloc[:, [0,1,2,3,4,5,6,8,9]]
y_train = train.iloc[:, 7]

X_test = test.iloc[:, [0,1,2,3,4,5,6,8,9]]
y_test = test.iloc[:, 7]


# Primero trabajamos con el TARGET 

# Clases unicas

classes = np.unique(y_train)
nclasses = len(classes)
print(f'classes: {list(classes)}')

# Generamos un encode para cada una de las clases. 

for i in range(0,len(classes)):
    y_train = y_train.replace(f'{list(classes)[i]}', i)
    y_test = y_test.replace(f'{list(classes)[i]}', i)
    
# Generamos un histograma para entender la distribución de las variables target 

plt.hist(y_train, bins=np.arange(y_train.min(), y_train.max()+2), density = True)

# ver value counts 
print(f'value counts: \n{y_train.value_counts()}')

# class weights 

from sklearn.utils import class_weight
import numpy as np

classes = np.unique(y_train)
class_weights_array = class_weight.compute_class_weight('balanced', classes = classes, y = y_train)
class_weights = {classes[i]: class_weights_array[i] for i in range(0, len(classes))}
class_weights

# onehot encoding target

y_train = keras.utils.to_categorical(y_train, nclasses)
y_test = keras.utils.to_categorical(y_test, nclasses)

# EDA: Ahora empezamos a trabajar con X

pd.set_option('display.float_format',lambda x:'%.3f'% x) #remove scientific notation

X_train.describe()

# distribucion 
sns.displot(
  data=X_train,
  x="cont_3",
  kind="hist",
  aspect=1.4,
  log_scale=10,
  bins=10
)

from sklearn.preprocessing import StandardScaler

#Normalization 

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

#scaler

scaler = StandardScaler()
scaler.fit(X_train[numeric_features])

X_train[numeric_features]  = scaler.transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

cols_common = [X_train.columns[i] for i in range(len(X_train.columns))]

# Feature correlations 

plt.figure(figsize = (25,10))

sns.heatmap(X_train[cols_common].corr().round(3),
            vmin = -1,
            vmax = 1,
            annot = True,
            cmap = 'RdBu')

plt.savefig(f'correlations.jpg')
plt.show()


# Aplicar One-Hot Encoding a las columna 'cat_7' y dropear las de ID + fecha

X_train = pd.get_dummies(X_train, columns=['cat_7'])
X_train = X_train.drop(columns = ['date_2','key_1'])

# Aplicar One-Hot Encoding a las columna 'cat_7' y dropear las de ID + fecha

X_test = pd.get_dummies(X_test, columns=['cat_7'])
X_test = X_test.drop(columns = ['date_2','key_1'])

## Construcción de una NN con arquitectura feedforward sencilla 

input_dim = X_train.shape[1]

model = Sequential()
model.add(Dense(18, input_shape = (input_dim,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nclasses, activation='softmax'))

import tensorflow_addons as tfa

#Definición de:
# - Hiperparametros de entrenamiento (Learning Rate, Loss Function, Optimizer, Batch Size, Epochs)
# - Métrica objetivo

learn_rate = .001
loss = 'categorical_crossentropy'
opt = Adam(learning_rate= learn_rate)

model.compile(optimizer= opt,loss = loss, metrics = [tf.metrics.Recall(thresholds=0.2)]) #tfa.metrics.F1Score(4, average='weighted',threshold=0.3)])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

batch_size = 64
epochs = 20
VERBOSE = 1

training_history = model.fit(X_train.astype('float32'),
                             y_train,
                             validation_data = (X_test.astype('float32'),y_test),
                             batch_size = batch_size,
                             epochs = epochs,
                             callbacks=[early_stopping],
                             class_weight = class_weights,
                             verbose = VERBOSE)


#Validate with test 

y_pseudo_probabilities = model.predict(X_test.astype('float32'))
y_pred = np.argmax(y_pseudo_probabilities)
y_test_max = np.argmax(y_test)


def eval_model(training, model, test_X, test_y, field_name):
    """
    Model evaluation: plots, classification report
    @param training: model training history
    @param model: trained model
    @param test_X: features 
    @param test_y: labels
    @param field_name: label name to display on plots
    """
    ## Trained model analysis and evaluation
    f, ax = plt.subplots(2,1, figsize=(5,5))
    ax[0].plot(training.history['loss'], label="Loss")
    ax[0].plot(training.history['val_loss'], label="Validation loss")
    ax[0].set_title('%s: loss' % field_name)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Recall
    ax[1].plot(training.history['recall'], label="recall")
    ax[1].plot(training.history['val_recall'], label="Validation recall")
    ax[1].set_title('%s: recall' % field_name)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('recall')
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    
    
eval_model(training_history, model, y_pred, y_test_max, 'Metrics')

# Model evaluation

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

# Print metrics
print("Classification report")

test_pred = model.predict(X_test.astype('float32'))

test_pred = np.argmax(test_pred, axis=1)
test_truth = np.argmax(y_test, axis=1)

print(metrics.classification_report(test_truth, test_pred))


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

loss = 'categorical_crossentropy'

def create_model(optimizer='adam'):

    # create model
    model = Sequential(name = "Zenpli")
    model.add(Dense(18, input_shape = (input_dim,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nclasses, activation='softmax'))
    
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.metrics.Recall(thresholds=0.2)])
    return model

model_grid = KerasClassifier(build_fn=create_model)

from sklearn.model_selection import GridSearchCV

params = {
    'batch_size':  [200],
    'epochs': [30],
    'optimizer':['adam']
}


recall_scorer = make_scorer(recall_score, greater_is_better=True)

grid = GridSearchCV(model_grid,params, scoring=recall_scorer, cv=5)

model_grid = grid.fit(X_train.astype('float32'),y_train)

model_grid.best_params_

model_grid.best_estimator_.model.summary()

from keras.models import load_model

model.save('Zenpli_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')