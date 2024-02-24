import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
import os
from tqdm import tqdm

# 设置一些参数
batch_size = 128
num_epochs = 20
learning_rate = 0.0001

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 定义ResNet模型
model = resnet18(weights=None, num_classes=len(train_dataset.classes)).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 设置TensorBoard
writer = SummaryWriter(log_dir='./logs')

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = outputs.max(1)
        correct_predictions += (predictions == labels).sum().item()

    # 计算训练集上的准确率和平均损失
    accuracy = correct_predictions / len(train_loader.dataset)
    average_loss = running_loss / len(train_loader)

    # 将训练集的准确率和损失写入TensorBoard
    writer.add_scalar('Training/Accuracy', accuracy, epoch)
    writer.add_scalar('Training/Loss', average_loss, epoch)

    print('epoch: ' + str(epoch) + '   ---训练准确率: ' + str(accuracy))
    print('epoch: ' + str(epoch) + '   ---训练损失值: ' + str(accuracy))

    # 在每5个epoch保存一次模型
    if (epoch + 1) % 1 == 0:
        checkpoint_path = os.path.join('savepoint', f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

    # 在测试集上评估模型
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1)
            correct_predictions += (predictions == labels).sum().item()

    # 计算测试集上的准确率
    test_accuracy = correct_predictions / len(test_loader.dataset)
    print('epoch: ' + str(epoch) + '   ---测试准确率: ' + str(accuracy))

    # 将测试集的准确率写入TensorBoard
    writer.add_scalar('Testing/Accuracy', test_accuracy, epoch)

# 关闭TensorBoard写入器
writer.close()
