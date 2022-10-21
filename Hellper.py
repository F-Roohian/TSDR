import pickle
import pandas as pd
import numpy as np
import random
from random import Random


def save (path , data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load (path):
    with open(path,'rb') as file:
        return pickle.load(file)
        
def read_csv(path):
    df_dataset = pd.read_csv(path, header=None, sep='\t', encoding='utf-8').to_numpy()
    return df_dataset

def pairs_disease_drug():
    path_b = 'dis_drug.txt'
    m_b_t = pd.read_csv(path_b, header=None, sep='\t', encoding='utf-8').to_numpy()
    SEED=4
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(m_b_t)[0]):
        for j in range(np.shape(m_b_t)[1]):
            if int(m_b_t[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(m_b_t[i][j]) == 0:
                whole_negative_index.append([i, j])
    Random(SEED).shuffle(whole_negative_index)            
    len_neg=len(whole_positive_index)
    negative_sample_index=whole_negative_index[:len_neg]
    return whole_positive_index, negative_sample_index

def partition_dataset(data_path,percent_train,input_shape_N):
    data = h.load(data_path)
    SEED=4
    Random(SEED).shuffle(data) 

    x = [i[:-1] for i in data]
    y = [i[-1] for i in data]
    #train and validation and test 
    len_data=int(len(data)*percent_train)
    x=np.array(x)
    y=np.array(y)
    #train 
    print('full data',len(x))
    train_x=x[:len_data]
    train_y=y[:len_data]
    num_input_train=len(train_x)
    print('number_train :',num_input_train)
    #test 
    test_x=x[len_data:]
    test_y=y[len_data:]
    num_input_test=len(test_x)
    print('number_test:',num_input_test)
    test_x = test_x.reshape(num_input_test,input_shape_N)
    train_x = train_x.reshape(num_input_train,input_shape_N)
    return train_x, train_y, test_x, test_y


#plot
def plot_training(history):

    #plot
    history_dict = history.history
    print('loss:',min(history_dict['loss']),'accuracy:',max(history_dict['accuracy']),
          'epoch:', np.argmax(history_dict['accuracy']))

    # res= myModel.predict(x_test)
    #plot loss
    import matplotlib.pyplot as plt 
    history_dict = history.history
    loss_values = history_dict['loss']
    epochs = range(1, len(loss_values)+1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.title('Training  loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    #plot accuracy
    plt.clf()
    acc_values = history_dict['accuracy']
    plt.plot(epochs,acc_values,'bo',label='Teraining acc')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    return  plt.show()
