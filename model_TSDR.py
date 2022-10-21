import tensorflow as tf
from tensorflow import keras
import Hellper as h
import numpy as np
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score ,plot_roc_curve
from keras.layers import Dense, Dropout, Input, Activation, Conv1D, MaxPooling1D, Flatten
from keras import regularizers
from keras.models import Model 
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
import time

#load data
path_train_dataset_1=f'Input_TSDR_drug_dis.data'
path_train_dataset_2=f'Input_TSDR_drug_pubchem.data'
path_train_dataset_3=f'Input_TSDR_drug_target_domain.data'
path_train_dataset_4=f'Input_TSDR_drug_target_go.data'
path_train_dataset_BiowordVec=f'Input_TSDR_drugs_BiowordVec.data' 

#partition dataset
input_shape_1=1362
input_shape_2=1304
input_shape_3=2107
input_shape_4=5128
input_shape_BiowordVec=1444
train_x_1, train_y_1, test_x_1, test_y_1=h.partition_dataset(path_train_dataset_1, 0.9, input_shape_1)
print('1:',len(train_x_1[0]))
train_x_2, train_y_2, test_x_2, test_y_2=h.partition_dataset(path_train_dataset_2, 0.9, input_shape_2)
print('2:',len(train_x_2[0]))
train_x_3, train_y_3, test_x_3, test_y_3=h.partition_dataset(path_train_dataset_3, 0.9, input_shape_3)
print('3:',len(train_x_3[0]))
train_x_4, train_y_4, test_x_4, test_y_4=h.partition_dataset(path_train_dataset_4, 0.9, input_shape_4)
print('4:',len(train_x_4[0]))
train_x_BiowordVec, train_y_BiowordVec, test_x_BiowordVec, lable=h.partition_dataset(path_train_dataset_BiowordVec, 0.9, input_shape_BiowordVec)#BiowordVec
print('fast:',len(train_x_BiowordVec[0]))

#Model Training--------------------------------------------------------------------
#hyperparameter
optimizer=RMSprop(lr=0.0001)
regular=regularizers.l1(0.00001)
name_network ='TSDR'
activation='relu'
batch_size=16
num_epochs=30
num_neuron_cnn_layer1=256
num_neuron_cnn_layer2=128
num_neuron_dense_layer1=256
num_neuron_dense_layer2=128
num_compact_features=64

# define four sets of inputs
inputX=Input(shape=(1362,1))      
inputY=Input(shape=(1304,1))      
inputD=Input(shape=(2107,1))      
inputE=Input(shape=(5128,1))      
inputF=Input(shape=(1444,1))      

#Step1-extracting compact features
#CNN-1
x=Conv1D(num_neuron_cnn_layer1, 9, activation=activation)(inputX)
x=MaxPooling1D(2)(x)
x=Conv1D(num_neuron_cnn_layer2, 9, activation=activation)(x)
x=MaxPooling1D(2)(x)
x=Flatten()(x)
x=Dense(num_neuron_dense_layer1, activation=activation)(x)
x=Dropout(0.2)(x)
x=Dense(num_neuron_dense_layer2, activation=activation)(x)
x=Dropout(0.2)(x)
x=Dense(num_compact_features, activation=activation)(x)
x=Model(inputs=inputA, outputs=x)
 
#CNN-2
y=Conv1D(num_neuron_cnn_layer1, 9, activation=activation)(inputY)
y=MaxPooling1D(2)(y)
y=Conv1D(num_neuron_cnn_layer2, 9, activation=activation)(y)
y=MaxPooling1D(2)(y)
y=Flatten()(y)
y=Dense(num_neuron_dense_layer1, activation=activation)(y)
y=Dropout(0.2)(y)
y=Dense(num_neuron_dense_layer2, activation=activation)(y)
y=Dropout(0.2)(y)
y=Dense(num_compact_features, activation=activation)(y)
y=Model(inputs=inputB, outputs=y)

#CNN-3
d=Conv1D(num_neuron_cnn_layer1, 9, activation=activation)(inputD)
d=MaxPooling1D(2)(d)
d=Conv1D(num_neuron_cnn_layer2, 9, activation=activation)(d)
d=MaxPooling1D(2)(d)
d=Flatten()(d)
d=Dense(num_neuron_dense_layer1, activation=activation)(d)
d=Dropout(0.2)(d)
d=Dense(num_neuron_dense_layer2, activation=activation)(d)
d=Dropout(0.2)(d)
d=Dense(num_compact_features, activation=activation)(d)
d=Model(inputs=inputD, outputs=d)

#CNN-4
e=Conv1D(num_neuron_cnn_layer1, 9, activation=activation)(inputE)
e=MaxPooling1D(2)(e)
e=Conv1D(num_neuron_cnn_layer2, 9, activation=activation)(e)
e=MaxPooling1D(2)(e)
e=Flatten()(e)
e=Dense(num_neuron_dense_layer1, activation=activation)(e)
e=Dropout(0.2)(e)
e=Dense(num_neuron_dense_layer2, activation=activation)(e)
e=Dropout(0.2)(e)
e=Dense(num_compact_features, activation=activation)(e)
e=Model(inputs=inputE, outputs=e)

#Step2 - combine the output of the four branches and classification
combined=tf.keras.layers.concatenate([x.output, y.output, d.output, e.output])
combined=tf.keras.layers.Reshape(( num_compact_features*4, 1))(combined)
added=concatenate([combined, inputF],axis=1)
g=Conv1D(num_neuron_cnn_layer1, 9, kernel_regularizer=regular, activation=activation)(added)
g=MaxPooling1D(2)(g)
g=Conv1D(num_neuron_cnn_layer2, 9, kernel_regularizer=regular, activation=activation)(g)
g=MaxPooling1D(2)(g)
g=Flatten()(g)
g=Dense(num_neuron_dense_layer1, kernel_regularizer=regular, activation=activation)(g)
g=Dropout(0.2)(g)
g=Dense(num_neuron_dense_layer2, kernel_regularizer=regular, activation=activation)(g)
g=Dropout(0.2)(g)
g=Dense(num_compact_features, kernel_regularizer=regular, activation=activation)(g)
g=Dense(1, activation="sigmoid")(g)

model=Model(inputs=[inputX, inputY, inputD ,inputE, inputF], outputs=g)
#compile
model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=["accuracy"])

#Training time 
start=time.time()
history=model.fit([train_x_1,train_x_2,train_x_3,train_x_4,train_x_BiowordVec], 
                    train_y_1,batch_size=batch_size, epochs=num_epochs, shuffle=True)
stop=time.time()
print(f"Training time: {(stop - start)/3600}")

#saving model
save_name_model='TSDR.h5'
model.save(save_name_model)
h.plot_training(history)

from keras.utils import plot_model
plot_model(model, to_file='TSDR.pdf', show_shapes=True)

#evaluate-test --------------------------------------------------------------------
result=model.evaluate([test_x_1, test_x_2, test_x_3, test_x_4, test_x_BiowordVec],lable)
predict=model.predict([test_x_1, test_x_2, test_x_3, test_x_4, test_x_BiowordVec])

#AUC
auc=sklearn.metrics.roc_auc_score(lable,predict)
print('AUC:',auc )

#AUPR
pr=tf.keras.metrics.PrecisionAtRecall(0.5)
pr.update_state(lable, predict)
precision_recall=pr.result().numpy()
print('precision_recall:',precision_recall)

#Precision
p=tf.keras.metrics.Precision()
p.update_state(lable, predict)
precision=p.result().numpy()
print('precision:',precision)

#Recall
r=tf.keras.metrics.Recall()
r.update_state(lable, predict)
recall=r.result().numpy()
print('recall:',recall)

#F1
F1=2 * (precision * recall) / (precision + recall)
print('F1:',F1)

