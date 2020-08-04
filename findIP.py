#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


Dataset = pd.read_csv('access_log_csv.csv')
print(Dataset)
NewDataset = Dataset.drop(['Log Name' , 'Time Zone' , 'Method' , 'Referer' , 'Bytes Sent', 'User Agent'], axis=1)
print(NewDataset)
X = NewDataset.iloc[:,:]
x = X.to_numpy()
print(x)
label = LabelEncoder()
IP = label.fit_transform(x[:,0])
print(IP)
Date = label.fit_transform(x[:,1])
print(Date)
URL = label.fit_transform(x[:,2])
RC = label.fit_transform(x[:,3])
df1 = pd.DataFrame(IP, columns=['IP'])
df2 = pd.DataFrame(Date, columns=['DATE'])
df3 = pd.DataFrame(URL, columns=['URL'])
df4 = pd.DataFrame(RC, columns=['Response Code'])
frames = [df1, df2, df3, df4]
result = pd.concat(frames, axis=1 )
print(result)

#Feature Normalization / Standardization using StandardScalar class
sc = StandardScaler()
data_scaled = sc.fit_transform(result)
print(data_scaled)
model = KMeans(n_clusters=10)
pred  = model.fit_predict(data_scaled)
Dataset_scaled = pd.DataFrame(data_scaled, columns=['IP', 'Date', 'URL', 'Response Code'])
Dataset_scaled['mycluster'] = pred
ips = [Dataset['Host'], result['IP']]
ips_result = pd.concat(ips, axis=1)
print(ips_result)
def CountFrequency(my_list, ip_label): 
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
    max_freq = 0
    max_key = 0
    for key, value in freq.items(): 
        if value > max_freq:
            max_freq = value
            max_key = key
    return ip_label[my_list.index(max_key)]
res = CountFrequency(ips_result['IP'].tolist(), ips_result['Host'].tolist())
res = str(res)
file1 = open("./suspectedIP.txt","w")
file1.write(res)
file1.close()
print("Suspicious IP is : ", res)