import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import init
import torch.nn.functional as F
from LossFunction import FocalLoss, SupContrastiveLoss
from Metrics import calculate_metrics
from DataPreprocessing import X_train, X_test, y_train, y_test
from OverSampling import X_train_over, y_train_over, X_train_total, y_train_total
import warnings

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
train_iter = Data.DataLoader(train_dataset, batch_size=4096, shuffle=True)


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


class MLP_Classfier(nn.Module):
    def __init__(self):
        super(MLP_Classfier, self).__init__()
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
        self.NormedLinearclassfier = NormedLinear(feat_dim=self.num_hiddens_3, num_classes=self.num_outputs)
        self.DisAlignLinearclassfier = DisAlignLinear(feat_dim=self.num_hiddens_3, num_classes=self.num_outputs, use_norm=False)
        self.LearnableWeightScalingLinearclassfier = LearnableWeightScalingLinear(feat_dim=self.num_hiddens_3, num_classes=self.num_outputs, use_norm=False)

    def forward(self, X):
        device_mlp = (X.device
                      if X.is_cuda
                      else torch.device('cpu'))
        before_predict = self.mlp(X).to(device_mlp)
        predict = F.relu(self.LearnableWeightScalingLinearclassfier(before_predict))
        return predict


mlp = MLP_Classfier().to(device)
# 初始化参数
for params in mlp.parameters():
    init.normal_(params, mean=0, std=0.01)

# 定义优化器
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-4)

num_epochs = 200
list_acc = []

# 训练模型
for epoch in range(1, num_epochs + 1):

    train_loss_tatal, train_acc_sum, n = 0.0, 0.0, 0
    test_acc_sum = 0.0

    for data, label in train_iter:
        # 如果有gpu，就使用gpu加速
        data = data.to(device)
        label = label.to(device)

        output = mlp(data)

        # 定义损失函数
        loss_1 = torch.nn.CrossEntropyLoss()
        # loss_2 = FocalLoss(num_class=5)
        loss_3 = torch.nn.MultiMarginLoss()
        batch_loss = loss_1(output, label)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        train_loss_tatal += batch_loss.item()
        train_acc_sum += (output.argmax(dim=1) == label).sum().item()
        n += label.shape[0]

    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        output = mlp(X_test)
        expected = y_test.cpu().numpy()
        predicted = output.argmax(dim=1).cpu().numpy()
        cm, accuracy, recall, precision, f1, classification_report = calculate_metrics(expected, predicted)
        # print(accuracy)
        test_acc_sum = (output.argmax(dim=1) == y_test).sum().item()

    epoch_loss_train = train_loss_tatal / n
    epoch_acc_train = train_acc_sum / n
    epoch_accuracy_test = test_acc_sum / y_test.shape[0]

    print('epoch %d, train loss %.6f,  train acc %.4f, test acc %.4f, Learning Rate: %.6f'
          % (epoch, epoch_loss_train, epoch_acc_train, epoch_accuracy_test, optimizer.param_groups[0]['lr']))
    print(f'\t\taccuracy: {accuracy:.4f}, recall: {recall:.4f}, precision: {precision:.4f}, f1: {f1:.4f}')
    list_acc.append(epoch_accuracy_test)

print(f'Max Test_Acc score: {max(list_acc):.4f}')
