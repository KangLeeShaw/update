import numpy as np
import glob
import utilities as UT
import pandas as pd
from  sklearn.model_selection import StratifiedKFold,KFold
LABEL_PATH='./data/'

label=[]
for i in range(0,546):
    label.append(0)
for i in range(0,734):
    label.append(1)
path=glob.glob('./data/AD2/*.jpg')
path2=glob.glob('./data/CN2/*.jpg')
#print(len(path))
#print(len(path2))
path3=path2+path
#print(path3)
#print(len(path3))
#print(label)
#print(len(label))
#random.shuffle(path3)
a=[]
path3=np.array(path3)
#path3.tolist()
#print(path3)
#print(path3)
kf=KFold(n_splits=5)
kf2=StratifiedKFold(n_splits=5,shuffle=True)
i=0

for train,test in kf2.split(path3,label):
    #print(test)
    #print(test.shape)
    example=path3[test]
    example=list(example)
    print(example)

    FILENAME = LABEL_PATH + '/fold_' + str(i) + '.csv'
    UT.write_csv(FILENAME,example)




    #FILENAME = LABEL_PATH + '/fold_' + str(i) + '.csv'
    df = pd.read_csv(FILENAME,header=None)

    data = df.values
    data = list(map(list,zip(*data)))
    data = pd.DataFrame(data)
    data.to_csv(FILENAME,header=0,index=0)
    i = i + 1










    #print(example)
    #print(example)

filenames=[LABEL_PATH + '/fold_0.csv',
           LABEL_PATH + '/fold_1.csv',
           LABEL_PATH + '/fold_2.csv',
           LABEL_PATH + '/fold_3.csv',
           LABEL_PATH + '/fold_4.csv']
with open(LABEL_PATH + '/ADEMCI_train_list.csv', 'w') as combined_train_list:
    for fold in filenames:
        for line in open(fold, 'r'):
            combined_train_list.write(line)

'''
a=UT.read_csv(LABEL_PATH+'/combined_train_list.csv')

full_path = a[0][0]

import cv2 as cv
img = cv.imread(full_path, 1)
cv.namedWindow('IMG')
cv.imshow("IMG", img)

cv.waitKey()
cv.destroyAllWindows()
'''