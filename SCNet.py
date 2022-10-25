import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import init
import torch.nn.functional as F
from LossFunction import FocalLoss, SupContrastiveLoss, ContrastiveSupLoss
from Metrics import calculate_metrics
from DataPreprocessing import X_train, X_test, y_train, y_test
from OverSampling import X_train_over, y_train_over, X_train_total, y_train_total
import warnings
import matplotlib.pyplot as plt
import os
import numpy as np
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

# 如果有gpu，就调用gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置seed
seed = 0
# 生成随机数，以便固定后续随机数，方便复现代码
random.seed(seed)
# 没有使用GPU的时候设置的固定生成的随机数
np.random.seed(seed)
# 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.manual_seed(seed)
# torch.cuda.manual_seed()为当前GPU设置随机种子
torch.cuda.manual_seed(seed)

y_train = torch.from_numpy(y_train)
# print(len(y_train))  # 125972
X_train = torch.from_numpy(X_train).float()
y_test = torch.from_numpy(y_test)
X_test = torch.from_numpy(X_test).float()

# 将数据集打包成DataLoader
train_dataset = Data.TensorDataset(X_train, y_train)
train_dataset.data = train_dataset.tensors[0]
train_dataset.targets = train_dataset.tensors[1]

# 将数据集打包成DataLoader
test_dataset = Data.TensorDataset(X_test, y_test)
test_dataset.data = test_dataset.tensors[0]
test_dataset.targets = train_dataset.tensors[1]
labels = ['dos' 'normal' 'probe' 'r2l' 'u2r']

train_dataset.classes = labels
test_dataset.classes = labels

train_dataset.classes_to_idx = {i: label for i, label in enumerate(labels)}
test_dataset.classes_to_idx = {i: label for i, label in enumerate(labels)}
train_iter = Data.DataLoader(train_dataset, batch_size=500, shuffle=True)


# 感知机
# 定义模型
class NormedLinear(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


class LearnableWeightScalingLinear(nn.Module):
    def __init__(self, feat_dim, num_classes, use_norm=False):
        super().__init__()
        self.classifier = NormedLinear(feat_dim, num_classes) if use_norm else nn.Linear(feat_dim, num_classes)
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.classifier(x) * self.learned_norm


class DisAlignLinear(nn.Module):
    def __init__(self, feat_dim, num_classes, use_norm=False):
        super().__init__()
        self.classifier = NormedLinear(feat_dim, num_classes) if use_norm else nn.Linear(feat_dim, num_classes)
        self.learned_magnitude = nn.Parameter(torch.ones(1, num_classes))
        self.learned_margin = nn.Parameter(torch.zeros(1, num_classes))
        self.confidence_layer = nn.Linear(feat_dim, 1)
        torch.nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, x):
        output = self.classifier(x)
        confidence = self.confidence_layer(x).sigmoid()
        return (1 + confidence * self.learned_magnitude) * output + confidence * self.learned_margin


class MLP_ConClassfier(nn.Module):
    def __init__(self):
        super(MLP_ConClassfier, self).__init__()
        self.num_inputs, self.num_hiddens_1, self.num_hiddens_2, self.num_hiddens_3, self.num_outputs \
            = 41, 512, 128, 32, 5
        self.num_proj_hidden = 32

        self.mlp_conclassfier = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_hiddens_1),
            nn.ReLU(),
            nn.Linear(self.num_hiddens_1, self.num_hiddens_2),
            nn.ReLU(),
            nn.Linear(self.num_hiddens_2, self.num_hiddens_3),
        )
        self.fc1 = torch.nn.Linear(self.num_hiddens_3, self.num_proj_hidden)
        self.fc2 = torch.nn.Linear(self.num_proj_hidden, self.num_hiddens_3)
        self.linearclassfier = nn.Linear(self.num_hiddens_3, self.num_outputs)
        self.NormedLinearclassfier = NormedLinear(feat_dim=self.num_hiddens_3, num_classes=self.num_outputs)
        self.DisAlignLinearclassfier = DisAlignLinear(feat_dim=self.num_hiddens_3, num_classes=self.num_outputs,
                                                      use_norm=True)
        self.LearnableWeightScalingLinearclassfier = LearnableWeightScalingLinear(feat_dim=self.num_hiddens_3,
                                                                                  num_classes=self.num_outputs,
                                                                                  use_norm=True)

    def projection_head(self, z):
        z = F.relu(self.fc1(z))
        return self.fc2(z)

    def forward(self, X):
        device_mlp = (X.device
                      if X.is_cuda
                      else torch.device('cpu'))
        before_predict = self.mlp_conclassfier(X).to(device_mlp)
        z = self.projection_head(before_predict)
        predict = F.relu(self.linearclassfier(before_predict))
        return predict, z


model = MLP_ConClassfier().to(device)
# 初始化参数
for params in model.parameters():
    init.normal_(params, mean=0, std=0.01)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=0.001)

num_epochs = 2000
list_acc = []

# 训练模型
for epoch in range(1, num_epochs + 1):

    train_loss_tatal, train_acc_sum, n = 0.0, 0.0, 0
    test_acc_sum = 0.0

    for data, label in train_iter:
        # 如果有gpu，就使用gpu加速
        data = data.to(device)
        label = label.to(device)

        output, z = model(data)

        # 定义损失函数
        loss_1 = torch.nn.CrossEntropyLoss()
        loss_2 = FocalLoss(num_class=5)
        # loss_3 = torch.nn.MultiMarginLoss()
        loss_4 = ContrastiveSupLoss(tau=1, device=device)
        batch_loss = loss_1(output, label) + 0.01 * loss_4(z, label)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        train_loss_tatal += batch_loss.item()
        train_acc_sum += (output.argmax(dim=1) == label).sum().item()
        n += label.shape[0]

    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        output, _ = model(X_test)
        expected = y_test.cpu().numpy()
        predicted = output.argmax(dim=1).cpu().numpy()
        cm, accuracy, recall, precision, f1, classification_report = calculate_metrics(expected, predicted)
        # print(accuracy)
        test_acc_sum = (output.argmax(dim=1) == y_test).sum().item()

    epoch_loss_train = train_loss_tatal / n
    epoch_acc_train = train_acc_sum / n
    epoch_accuracy_test = test_acc_sum / y_test.shape[0]
    scheduler.step(epoch_loss_train)

    print('epoch %d, train loss %.6f,  train acc %.4f, test acc %.4f, Learning Rate: %.6f'
          % (epoch, epoch_loss_train, epoch_acc_train, epoch_accuracy_test, optimizer.param_groups[0]['lr']))
    print(f'\t\taccuracy: {accuracy:.4f}, recall: {recall:.4f}, precision: {precision:.4f}, f1: {f1:.4f}')
    list_acc.append(epoch_accuracy_test)
    print(f'\t\tCurrent Max Test_Acc score: {max(list_acc):.4f}')
    # 模型保存
    if accuracy >= max(list_acc):  # 如果成绩高，保存模型
        torch.save(model.state_dict(), './checkpoint/SCNet.pth')

print(f'Total Max Test_Acc score: {max(list_acc):.4f}')
plt.plot(torch.arange(1, (len(list_acc)) + 1), list_acc)
plt.show()
