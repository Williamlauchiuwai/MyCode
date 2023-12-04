#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import shutil



def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        print("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


class Net(nn.Module):
    def __init__(self,n_featuers,n_hidden,n_outputs):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(n_featuers,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,n_outputs)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


#----------------------------------------------------------------------
def calc_acc(loader=None, net=None, loss_func=None):
    """"""
    

    test_loss = 0
    correct = 0
    for data, target in loader:
        data = data.view(-1, 28 * 28)    # 展平数据集，-1表示自适应

        logits = net(data)
        test_loss += loss_func(logits, target).item()

        pred = logits.argmax(dim=1)          # 模型的预测，argmax是对所有输出结果的概率求最大值
        correct += pred.eq(target).float().sum().item()

    test_loss /= len(loader.dataset)
    print('\n set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))     # 输出模型的平均loss和准确率
    
    # 在测试集上计算准确度
    total_num = len(loader.dataset)
    acc = correct / total_num
    print('set acc:', acc)  
    
    return acc
    



if __name__ == '__main__':
 
    # record loss and acc
    
    # 数据集构建
    if os.path.exists('mnist'):
        pass
    else:
        shutil.rmtree('minst') if os.path.exists('mnist') else 1
    writer = SummaryWriter('mnist', comment='mnist')
    
    batch_size=200       # 每个batch大小为200
    learning_rate=0.01   # 学习率
    epochs=5            # 共计10个epochs
    
 
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)
    
    
 
    
  
    x, y = next(iter(train_loader))
    print(x.shape, y.shape, x.min(), x.max())
    plot_image(x, y, 'image sample')
    
 
    # 模型构建
    input_dim = 28*28
    hidden_dim = 64
    out_dim = 10
    
    net = Net(input_dim,hidden_dim,out_dim)
    
    # 优化器构建
    optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    
    train_record_step = 0
    test_record_step = 0
    
    # 模型训练
    for epoch in range(epochs):
        net.train()
        for step,(data_x,data_y) in enumerate(train_loader):
            train_record_step += 1
            data = data_x.view(-1,28*28)   # 将28*28的矩阵展平为784长度的向量，作为输入层，-1表示自适
    
            prediction = net(data)
            loss = loss_func(prediction,data_y) # data_y表示数据的标签,在CrossEntropyLoss中被表示成one-hot向量的形式
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if step % 100 == 0:            # 每100个batch，打印一下训练结果
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, step * len(data), len(train_loader.dataset),
                           100. * step / len(train_loader), loss.item()))
                
            writer.add_scalar('train_loss',loss,global_step=train_record_step)
            
        train_acc = calc_acc(loader=train_loader, net=net, loss_func=loss_func)
       
        # 模型评测
        with torch.no_grad():
            net.eval()
            for step,(data_x,data_y) in enumerate(test_loader):
                test_record_step += 1
                data = data_x.view(-1,28*28)   # 将28*28的矩阵展平为784长度的向量，作为输入层，-1表示自适
        
                prediction = net(data)
                loss = loss_func(prediction,data_y) # data_y表示数据的标签,在CrossEntropyLoss中被表示成one-hot向量的形式
        
        
                if step % 100 == 0:            # 每100个batch，打印一下训练结果
                    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, step * len(data), len(test_loader.dataset),
                               100. * step / len(test_loader), loss.item()))  
                    
                writer.add_scalar('test_loss',loss,global_step=test_record_step)
                
        test_acc = calc_acc(loader=test_loader, net=net, loss_func=loss_func)           
        writer.add_scalars('acc',{'train_acc':train_acc,'test_acc':test_acc},global_step=epoch)
            
    

 
    
    # 结果显示
    x, y = next(iter(test_loader))
    out = net(x.view(x.size(0), 28*28))
    pred = out.argmax(dim=1)
    plot_image(x, pred, 'predict')
    plt.show()
 