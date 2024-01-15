import numpy as np
import pandas as pd
import csv

dataset= pd.read_csv('data.csv')

def entropy(target_col): # entory caluclation
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy
  

def InfoGain(data,split_attribute_name,target_name="Decision"): #information gain 
    total_entropy = entropy(data[target_name])
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    InfoGain = total_entropy - Weighted_Entropy
    return InfoGain

    


def ID3(data,originaldata,features,target_attribute_name="Decision",parent_node_class = None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    elif len(features) ==0:
        return parent_node_class
    else:
            parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
           #calculate information gain
            item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
            best_feature_index = np.argmax(item_values) # select highest gain attribute 
            best_feature = features[best_feature_index]
            tree = {best_feature:{}} #root of the tree
            features = [i for i in features if i != best_feature] # subtree
            for value in np.unique(data[best_feature]):
                value = value
                sub_data = data.where(data[best_feature] == value).dropna()
                subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
                tree[best_feature][value] = subtree # assign sub tree to root
            return(tree)
    
tree = ID3(dataset,dataset,dataset.columns[:-1])
    #print(dataset.head())
print(' \nDisplay Tree\n',tree)