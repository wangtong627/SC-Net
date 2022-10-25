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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings("ignore")

# 如果有gpu，就调用gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

y_train = torch.from_numpy(y_train)
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
train_iter = Data.DataLoader(train_dataset, batch_size=10000, shuffle=True)


# 感知机
# 定义模型
class MLP_with_2_layer(nn.Module):
    def __init__(self):
        super(MLP_with_2_layer, self).__init__()
        self.num_inputs, self.num_hiddens_1, self.num_hiddens_2, self.num_outputs \
            = 41, 128, 32, 5
        self.mlp = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_hiddens_1),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            # nn.BatchNorm1d(self.num_hiddens_1),
            nn.Linear(self.num_hiddens_1, self.num_hiddens_2),
            # nn.Dropout(p=0.3),
        )
        self.linearclassfier = nn.Linear(self.num_hiddens_2, self.num_outputs)

    def forward(self, X):
        device_mlp = (X.device
                      if X.is_cuda
                      else torch.device('cpu'))
        before_predict = self.mlp(X).to(device_mlp)
        predict = F.relu(self.linearclassfier(before_predict))
        return predict, before_predict


class MLP_with_3_layer(nn.Module):
    def __init__(self):
        super(MLP_with_3_layer, self).__init__()
        self.num_inputs, self.num_hiddens_1, self.num_hiddens_2, self.num_hiddens_3, self.num_outputs \
            = 41, 512, 128, 32, 5
        self.mlp = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_hiddens_1),
            nn.ReLU(),
            nn.Linear(self.num_hiddens_1, self.num_hiddens_2),
            nn.ReLU(),
            nn.Linear(self.num_hiddens_2, self.num_hiddens_3),
        )
        self.linearclassfier = nn.Linear(self.num_hiddens_3, self.num_outputs)

    def forward(self, X):
        device_mlp = (X.device
                      if X.is_cuda
                      else torch.device('cpu'))
        before_predict = self.mlp(X).to(device_mlp)
        predict = F.relu(self.linearclassfier(before_predict))
        return predict, before_predict


class MLP_with_4_layer(nn.Module):
    def __init__(self):
        super(MLP_with_4_layer, self).__init__()
        self.num_inputs, self.num_hiddens_1, self.num_hiddens_2, self.num_hiddens_3, self.num_hiddens_4, self.num_outputs \
            = 41, 128, 64, 32, 16, 5
        self.mlp = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_hiddens_1),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            # nn.BatchNorm1d(self.num_hiddens_1),
            nn.Linear(self.num_hiddens_1, self.num_hiddens_2),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(self.num_hiddens_2, self.num_hiddens_3),
            nn.ReLU(),
            nn.Linear(self.num_hiddens_3, self.num_hiddens_4),
        )
        self.linearclassfier = nn.Linear(self.num_hiddens_4, self.num_outputs)

    def forward(self, X):
        device_mlp = (X.device
                      if X.is_cuda
                      else torch.device('cpu'))
        before_predict = self.mlp(X).to(device_mlp)
        predict = F.relu(self.linearclassfier(before_predict))
        return predict, before_predict


class MLP_with_5_layer(nn.Module):
    def __init__(self):
        super(MLP_with_5_layer, self).__init__()
        self.num_inputs, self.num_hiddens_1, self.num_hiddens_2, self.num_hiddens_3, self.num_hiddens_4, self.num_hiddens_5, self.num_outputs \
            = 41, 256, 128, 64, 32, 16, 5
        self.mlp = nn.Sequential(
            nn.Linear(self.num_inputs, self.num_hiddens_1),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            # nn.BatchNorm1d(self.num_hiddens_1),
            nn.Linear(self.num_hiddens_1, self.num_hiddens_2),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(self.num_hiddens_2, self.num_hiddens_3),
            nn.ReLU(),
            nn.Linear(self.num_hiddens_3, self.num_hiddens_4),
            nn.ReLU(),
            nn.Linear(self.num_hiddens_4, self.num_hiddens_5),
        )
        self.linearclassfier = nn.Linear(self.num_hiddens_5, self.num_outputs)

    def forward(self, X):
        device_mlp = (X.device
                      if X.is_cuda
                      else torch.device('cpu'))
        before_predict = self.mlp(X).to(device_mlp)
        predict = F.relu(self.linearclassfier(before_predict))
        return predict, before_predict


model = MLP_with_3_layer().to(device)
# 初始化参数
for params in model.parameters():
    init.normal_(params, mean=0, std=0.01)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=0.0005)

num_epochs = 10
list_acc = []

# 训练模型
for epoch in range(1, num_epochs + 1):

    train_loss_tatal, train_acc_sum, n = 0.0, 0.0, 0
    test_acc_sum = 0.0

    for data, label in train_iter:
        # 如果有gpu，就使用gpu加速
        data = data.to(device)
        label = label.to(device)

        output, before_predict = model(data)

        # 定义损失函数
        loss_1 = torch.nn.CrossEntropyLoss()
        # loss_2 = FocalLoss(num_class=5)
        loss_3 = torch.nn.MultiMarginLoss()
        loss_4 = ContrastiveSupLoss(tau=1, device=device)
        batch_loss = loss_1(output, label)  # + 0.1 * loss_4(before_predict, label)

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
    print(f'Current Max Test_Acc score: {max(list_acc):.4f}')
    # 模型保存
    if accuracy >= max(list_acc):  # 如果成绩高，保存模型
        torch.save(model.state_dict(), 'net_params_Rx.pth')

print(f'Total Max Test_Acc score: {max(list_acc):.6f}')
plt.plot(torch.arange(1, (len(list_acc)) + 1), list_acc)
plt.show()
