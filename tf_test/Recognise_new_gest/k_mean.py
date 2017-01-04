import math
import numpy as np

def k_mean_index(indices,num_classes):
    #takes the array of indices and return n_classes number of cluster means
    num_indices = len(indices)
    means = np.zeros(num_classes)
    new_means = np.zeros(num_classes)
    for i in range(num_classes):
        new_means[i] = indices[math.floor((i+0.5)*num_indices/num_classes)]
    while(not(np.allclose(new_means,means))):
        means = np.copy(new_means)
        member_total = np.zeros(num_classes)
        member_num = np.zeros(num_classes)
        for i in range(num_indices):
            index_class = (abs(indices[i] - means)).argmin()
            member_total[index_class]+=indices[i]
            member_num[index_class]+=1
        new_means = np.divide(member_total,member_num)
    return new_means

def find_indices(preds):
    indices = []
    for i in range(len(preds)):
        if preds[i]>0.9:
            indices.append(i)
    return np.array(indices)
