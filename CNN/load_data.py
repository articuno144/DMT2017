import numpy as np
import pandas as pd

cali_d = pd.read_csv('cali_d.csv',sep = ',', header = None).values

noise_dataframe = pd.read_csv('noise.csv',sep= ',', header= None)
noise_values = np.array(noise_dataframe.values[:,0],dtype = 'float')
noise_values = noise_values.reshape((6,300,-1))

noised_dataframe = pd.read_csv('noised.csv',sep= ',', header= None)
noised_values = noised_dataframe.values[:,0]
noised_values = np.transpose(noised_values.reshape((9,-1)))

def standardize_set(arr):
    a3 = np.multiply(np.add(arr[3,50::5],-250),0.01)
    a4 = np.multiply(np.add(arr[4,50::5],-250),0.01)
    a5 = np.multiply(np.add(arr[5,50::5],-250),0.01)
    return np.array([a3,a4,a5])

new_n = np.zeros([3,50,91000])
new_nt = np.zeros([3,50,18000])

for j in range(91000):
    new_n[:,:,j] = standardize_set(noise_values[:,:,j])
    
batch_size = 1000
noise_values = np.transpose(new_n,[2,1,0])
