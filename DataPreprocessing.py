# import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
# import seaborn as sns

"""
DATASET SOURCE is from https://github.com/arjbah/nsl-kdd.git (include the most attack types)
https://github.com/defcom17/NSL_KDD.git
"""

train_file = './data/KDDTrain+.txt'
test_file = './data/KDDTest+.txt'
field_name_file = './data/Field_Names.csv'
attack_type_file = './data/training_attack_types.txt'
# attack_type_file = './data/Attack Types.csv'


field_names_df = pd.read_csv(field_name_file, header=None, names=['name', 'data_type'])  # 定义dataframe ，并给个column name，方便索引
field_names = field_names_df['name'].tolist()
field_names += ['label', 'label_code']  # 源文件中没有标签名称，以及等级信息
df = pd.read_csv(train_file, header=None, names=field_names)
# label = set(df['label'])
df_test = pd.read_csv(test_file, header=None, names=field_names)
attack_type_df = pd.read_csv(attack_type_file, sep=' ', header=None, names=['name', 'attack_type'])
attack_type_dict = dict(
    zip(attack_type_df['name'].tolist(), attack_type_df['attack_type'].tolist()))  # 定义5大类和小类的映射字典，方便替代
df.drop('label_code', axis=1, inplace=True)  # 最后一列 既无法作为feature，也不是我们的label，删掉
df_test.drop('label_code', axis=1, inplace=True)
df['label'].replace(attack_type_dict, inplace=True)  # 替换label 为5 大类
df_test['label'].replace(attack_type_dict, inplace=True)
# print(df.info())


# 简单定义一个print 函数
def print_label_dist(label_col):
    c = Counter(label_col)
    print(f'label is {c}')
#
# print_label_dist(df['label'])
# print_label_dist(df_test['label'])


# train_label = df[['label']]
# train_label['type'] = 'train'
# test_label = df_test[['label']]
# test_label['type'] = 'test'
# label_all = pd.concat([train_label, test_label], axis=0)
# print(label_all)
# print(test_label)
# sns.countplot(x='label', hue='type', data=label_all)

# y = df['label']
# y_test = df_test['label']
# X = df.drop('label', axis=1)
# X_test = df_test.drop('label', axis=1)

data = pd.concat([df, df_test])  # (148517, 42)
# 获取特征和标签
labels = data.iloc[:, 41]  # (148517,)
data = data.drop(columns=['label'])  # (148517, 41)

# 标签编码
le = LabelEncoder()
labels = le.fit_transform(labels).astype(np.int64)
# print(le.classes_)
# print_label_dist(labels)

# 特征编码
data['protocol_type'] = le.fit_transform(data['protocol_type'])
data['service'] = le.fit_transform(data['service'])
data['flag'] = le.fit_transform(data['flag'])

# 标签和特征转成numpy数组
data = np.array(data)  # (144765,41)
labels = np.array(labels)  # (144765,)

# 特征值归一化
min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(data)

X_train, X_test, y_train, y_test = data[:125973], data[125973:], labels[:125973], labels[125973:]
# print_label_dist(y_train)


# # 转成torch.tensor类型
# labels_tensor = torch.from_numpy(labels)
# data_tensor = torch.from_numpy(data).float()
#
# X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = data_tensor[:125972], data_tensor[125972:],\
#                                                                labels_tensor[:125972], labels_tensor[125972:]
