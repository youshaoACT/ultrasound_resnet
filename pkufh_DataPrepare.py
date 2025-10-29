import pandas as pd
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split

f = pd.read_csv("/home/vipuser/ultrasound/radiomics/增强图像且扫片.csv",dtype = {"hadm_id":str})
fo = pd.read_csv('/home/vipuser/ultrasound/radiomics/pkufh_feature&outcome.csv',dtype = {"hadm_id":str})

f = f[["hadm_id","is_CR","is_SC"]]
fo = fo[['ID', 'hadm_id', 'group']]

f = pd.merge(f,fo,how='left',on='hadm_id')
f.drop(["group"],axis = 1,inplace = True)
f.dropna(inplace = True)


des_path = "/data_vdc/dataset_diffmic/images"
data_list = []
for line in f.iterrows():
    ID = line[1]['ID']
    image_path = os.path.join(des_path, ID)
    label = line[1]['is_CR']
    dic = {"image_root": image_path, "label": label}
    data_list.append(dic)

train_list, test_list = train_test_split(
    data_list, test_size=0.2, random_state=42,
    stratify=[item['label'] for item in data_list]
)


f_train = pd.DataFrame(train_list)
f_test = pd.DataFrame(test_list)

print(f"训练集样本数: {len(train_list)}")
print(f"测试集样本数: {len(test_list)}")



pkl_path_train = "/home/vipuser/ultrasound/resnet/CR_DataList_train.pkl"
with open(pkl_path_train,'wb') as f_pkl:
    pickle.dump(train_list,f_pkl)

pkl_path_test = "/home/vipuser/ultrasound/resnet/CR_DataList_test.pkl"
with open(pkl_path_test,'wb') as f_pkl:
    pickle.dump(test_list,f_pkl)
