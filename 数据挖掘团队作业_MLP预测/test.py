import numpy as np
import torch
import matplotlib.pyplot as plt

def evaluate(model_set, test_y):


    model_set = model_set.float()
    test_y = torch.from_numpy(test_y).float()

    num_test, _ = model_set.shape
    loss_fn = torch.nn.MSELoss(reduction='none')
    loss = torch.mean(loss_fn(model_set, test_y))
    
    print("Test:\n Loss: {}".format(loss))
    model_set = (model_set+0.5).int()
    test_y = test_y.int()

    sum = 0
    for i in range(num_test):
        if model_set[i] == test_y[i]:
            sum += 1

    print("{}%".format(sum/num_test *100))
    
    return(sum/num_test * 100)

def visualize(x, y):

    plt.figure(figsize=(10,5))#设置画布的尺寸
    plt.title('wine prediction',fontsize=20)#标题，并设定字号大小
    plt.xlabel(u'x-EPOCH',fontsize=14)#设置x轴，并设定字号大小
    plt.ylabel(u'y-ACC',fontsize=14)#设置y轴，并设定字号大小

    #color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    plt.plot(x, y,color="deeppink",linewidth=2,linestyle=':',label='Jay income', marker='o')
    # plt.plot(x, y,color="darkblue",linewidth=1,linestyle='--',label='JJ income', marker='+')

    plt.show()#显示图像



Rec_Train_Loss_path = 'Rec_Train_Loss.res.npy'
Rec_ACC_path = 'Rec_ACC.res.npy'
Rec_Epoch_path = 'Rec_Epoch.res.npy'

Rec_Train_Loss = np.load(Rec_Train_Loss_path)
Rec_ACC = np.load(Rec_ACC_path)
Rec_Epoch = np.load(Rec_Epoch_path)

visualize(Rec_Epoch, Rec_ACC)