from torch.nn import parameter
from model import MLP
import torch
import numpy as np

epochs = 10000
batch_size = 100

torch.random.seed = 2021

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



def train(train_x, train_y, test_x, test_y):
    Rec_Epoch = []
    Rec_ACC = []
    Rec_Train_Loss = []

    input_num, input_dim = train_x.shape
    _, output_dim = train_y.shape
    model = MLP(input_size=input_dim, common_size=output_dim)

    # parameter = list(model.parameter())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =epochs)
    loss_fn = torch.nn.MSELoss(reduction='none')

    # model.cuda()

    iteration = input_num // batch_size

    for i in range(epochs):
        permutation = list(np.random.permutation(input_num))  #m为样本数
        train_x = train_x[permutation]
        train_y = train_y[permutation]
        for j in range(11):
            train_batch = train_x[j*batch_size:(j+1)*batch_size]
            result_batch = train_y[j*batch_size:(j+1)*batch_size]
            output = model(train_batch)
            result_batch = torch.from_numpy(result_batch).float()
            loss = torch.mean(loss_fn(output, result_batch))
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (i+1)%100 == 0:
            print('------------------------')
            print('EPOCH:{}:'.format(i+1))
            print('Train:\n Loss : {}'.format(loss))
            model_set = model(test_x)
            ACC = evaluate(model_set, test_y)
            Rec_ACC.append(ACC)
            Rec_Epoch.append(i+1)
            Rec_Train_Loss.append(loss.item())
            print('------------------------')
            # torch.save(model, 'result/model_{}.pth'.format(i+1))
    Rec_Train_Loss = np.array(Rec_Train_Loss)
    Rec_ACC = np.array(Rec_ACC)
    Rec_Epoch = np.array(Rec_Epoch)
    np.save('Rec_Train_Loss.res',Rec_Train_Loss)
    np.save('Rec_ACC.res',Rec_ACC)
    np.save('Rec_Epoch.res',Rec_Epoch)






