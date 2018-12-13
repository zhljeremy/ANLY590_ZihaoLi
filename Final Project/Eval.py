# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:50:55 2018

@author: hongx
"""

import pandas as pd

def evaluation(path):
    output_df = pd.read_csv(path, header=None)
    
    true_value = list(output_df[102])
    prob_df = output_df.drop(labels=102, axis=1) 
    top5_list = []
    
    for row in prob_df.iterrows():
        index, data = row
        data = data.tolist()
        top5_list.append(
            sorted(range(len(data)), key=lambda i: data[i])[-10:][::-1])
        
    zip_list = zip(true_value,top5_list)
    check= list([int(x) == y[0] for x,y in zip_list])
    print("Top1 Accuracy: ", sum(check)/len(check))
    
    count = 0
    for x,y in zip(true_value,top5_list):
        if int(x) in y[:3]:
            count += 1
    
    print("Top3 Accuracy: ", count/len(top5_list))
    
    count = 0
    for x,y in zip(true_value,top5_list):
        if int(x) in y:
            count += 1
    
    print("Top5 Accuracy: ", count/len(top5_list))

evaluation('./output_MobileNet.csv')