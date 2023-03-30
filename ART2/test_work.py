from art2 import Art2Network
from utils_new import *
import pandas as pd

# number classification
x_train=[
    [1,1,1,0,1,1,1],
    [0,0,1,0,0,1,0],
    [1,0,1,1,1,0,1],
    [1,0,1,1,0,1,1],
    [0,1,1,1,0,1,0],
    [1,1,0,1,0,1,1],
    [1,1,0,1,1,1,1],
    [1,0,1,0,0,1,0],
    [1,1,1,1,1,1,1],
    [1,1,1,1,0,1,1]
]
y_train=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]

L1_size = 7
L2_size = 9

net = Art2Network(L1_size, L2_size, 0.95)
y_train_pred = net.process_points(x_train, True)
print(f'Test accuracy: {cluster_acc(y_train, y_train_pred)}')
show_confusion_matrix(y_train, y_train_pred)
# accuracy=0.5


# animals atribute classification
animal=pd.read_excel(io=r'C:\Users\pc\Desktop\大三の绝处逢生\计算智能\week4\Zoo database\Zoo database\data_transfer.xlsx')
aributes=animal.iloc[:,1:len(animal.columns)-1]
label=animal.iloc[:,len(animal.columns)-1]

atributes_train=aributes.values
label=label.values

target=[]
for i in range(len(label)):
    target.append([label[i]])

L1_size = len(animal.columns)-2
L2_size = 7

net2 = Art2Network(L1_size, L2_size, 0.95)
target_pred = net2.process_points(atributes_train, True)
print(f'Test accuracy: {cluster_acc(target, target_pred)}')
show_confusion_matrix(target, target_pred)
# accuracy=0.66