# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：KD.py
@时间：2023/3/3  10:42
@文档说明:
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

# 设置随机种子，便于复现
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 使用cuDNN加速卷积运算
torch.backends.cudnn.benchmark = True
# 载入MNIST数据集
train_dataset = torchvision.datasets.MNIST(
    root = "datasets/",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
test_dataset = torchvision.datasets.MNIST(
    root = "datasets/",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)
train_loader = DataLoader(dataset = train_dataset,batch_size = 32,shuffle = True)
test_loader = DataLoader(dataset = test_dataset,batch_size = 32,shuffle = True)
# 构建教师网络
class TeacherModel(nn.Module):
    def __init__(self,in_channels = 1,num_class = 10):
        super(TeacherModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784,1200)
        self.fc2 = nn.Linear(1200,1200)
        self.fc3 = nn.Linear(1200,num_class)
        self.dropout = nn.Dropout(p = 0.5)
    def forward(self,x):
        x = x.view(-1,784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x
# 从头开始训练教师模型
model = TeacherModel()
model = model.to(device)
summary(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)
epochs = 6
# for epoch in range(epochs):
#     model.train()
#
#     for data,targets in tqdm(train_loader):
#         data = data.to(device)
#         targets = targets.to(device)
#         preds = model(data)
#         loss = criterion(preds,targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     num_correct = 0
#     num_sample = 0
#     with torch.no_grad():
#         for x, y  in test_loader:
#             x= x.to(device)
#             y = y.to(device)
#             preds = model(x)
#             preditions = preds.max(1).indices
#             num_correct +=(preditions==y).sum()
#             num_sample +=preditions.size(0)
#             acc = (num_correct/num_sample).item()
#     model.train()
#     print('Epoch:{}\t Accuracy:{:4f}'.format(epoch+1,acc))

teacher_model = model

# 构建学生模型,中间的线性层由1200转变为20
class StudentModel(nn.Module):
    def __init__(self,in_channel = 1, num_class = 10):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3=nn.Linear(20,num_class)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,x):
        x= x.view(-1,784)
        x= self.fc1(x)
        x= self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x
# 单独训练学生模型，并给出结果
model = StudentModel()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)
epochs = 3
for epoch in range(epochs):
    model.train()
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        preds = model(data)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    num_correct = 0
    num_sample = 0
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            preditions = preds.max(1).indices
            num_correct += (preditions==y).sum()
            num_sample +=preditions.size(0)
        acc = (num_correct/num_sample).item()
    model.train()
    print("Epoch:{}\t Accuracy:{:.4f}".format(epoch+1,acc))

student_model_scartch = model

# 准备好预训练的教师模型
teacher_model.eval()

model = StudentModel()
model = model.to(device)
model.train()

# 设置蒸馏温度
temp = 7

# 设置hard_loss
hard_loss = nn.CrossEntropyLoss()
alpha = 0.3
# 设置soft_loss
soft_loss = nn.KLDivLoss(reduction = "batchmean")
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
# 通过蒸馏训练学生模型
epochs = 3
for epoch in range(epochs):
    # 载入训练数据,进行蒸馏训练
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # 教师模型预测
        with torch.no_grad():
            teacher_pred = teacher_model(data)
        # 学生模型预测
        student_pred = model(data)
        # 计算hard_loss
        student_loss = hard_loss(student_pred,targets)
        # 计算soft_loss
        ditillation_loss = soft_loss(F.softmax(student_pred/temp,dim = 1),
                                     F.softmax(teacher_pred/temp,dim = 1))
        # 计算最终的loss
        loss = alpha * student_loss+(1-alpha)*ditillation_loss
        # 反向传播，进行权重更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 测试集上评估性能
    model.eval()
    num_correct = 0
    num_sample = 0
    with torch.no_grad():

        for data ,targets in test_loader():
            data = data.to(device)
            targets = targets.to(device)
            preds = model(x)
            preditions =preds.max(1).indices
            num_correct += (preditions==targets).sum()
            num_sample += preditions.size(0)
        acc = (num_correct/num_sample).item()
    model.train()
    print("Epoch:{}\t Accuracy:{.4f}".format(epoch+1,acc))






