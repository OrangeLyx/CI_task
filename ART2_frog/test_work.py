from art2 import Art2Network
from utils_new import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#  frog classification
frog=pd.read_excel(io=r'C:\Users\pc\Desktop\大三の绝处逢生\计算智能\week5\frog_data.xlsx')
frog=frog.sample(frac=1)

label=frog['RecordID'].tolist()

target=[]
for i in range(len(label)):
    target.append([label[i]])

le=LabelEncoder()
np.set_printoptions(threshold=np.inf)
frog['Family']=le.fit_transform(frog['Family'])
frog['Genus']=le.fit_transform(frog['Genus'])
frog['Species']=le.fit_transform(frog['Species'])
atributes=frog.iloc[:,:len(frog.columns)-1]
atributes=atributes.values


L1_size = 25
L2_size = 60

net = Art2Network(L1_size, L2_size, 0.9)
target_pred = net.process_points(atributes, True)
print(f'Test accuracy: {cluster_acc(target, target_pred)}')
show_confusion_matrix(target, target_pred)
# accuracy=