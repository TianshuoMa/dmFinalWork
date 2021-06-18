import pandas as pd
from train import *


data_train_name = r'data\data_train.csv'
target_train_name = r'data\target_train.csv'
data_test_name = r'data\data_test.csv'
target_test_name = r'data\target_test.csv'



# file = pd.read_csv(data_name)
# data = pd.DataFrame(file)

file_data_train = pd.read_csv(data_train_name)
# data_train = pd.DataFrame(file_data_train)

file_data_test = pd.read_csv(data_test_name)
# data_test = pd.DataFrame(file_data_test)

file_target_train = pd.read_csv(target_train_name)
# target_train  = pd.DataFrame(file_target_train).values.flatten()

file_target_test = pd.read_csv(target_test_name)
# target_test = pd.DataFrame(file_target_test).values.flatten()



train_x = file_data_train.values
train_y = file_target_train.values
test_x  = file_data_test.values
test_y = file_target_test.values


train(train_x,train_y,test_x,test_y)
# evaluate(test_x, test_y)


