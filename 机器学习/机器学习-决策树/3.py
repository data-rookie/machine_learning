import torch
from torch import nn
# create training dataset
train_dataset=[[0, 0, 0, 0, 0, 0, 1],[1, 0, 1, 0, 0, 0, 1],
               [1, 0, 0, 0, 0, 0, 1],[0, 0, 1, 0, 0, 0 ,1],
               [2, 0, 0, 0, 0, 0, 1],[0, 1, 0, 0, 1, 1, 1],
               [1, 1, 0, 1, 1, 1, 1],[1, 1, 0, 0, 1, 0, 1],
               [1, 1, 1, 1, 1, 0, 0],[0, 2, 2, 0, 2, 1, 0],
               [2, 2, 2, 2, 2, 0, 0],[2, 0, 0, 2, 2, 1, 0],
               [0, 1, 0, 1, 0, 0, 0],[2, 1, 1, 1, 0, 0, 0],
               [1, 1, 0, 0, 1, 1, 0],[2, 0, 0, 2, 2, 0, 0],
               [0, 0, 1, 1, 1, 0, 0]
]


def one_hot(input_data):
    # load data into torch and change the data's dimension(turn M×n into n×M)
    input_data_copy = torch.tensor(input_data).t()
    # get the numbers of data and data's feature
    feature_num, data_num = input_data_copy.shape
    # create a tenosr to save output
    output_X = torch.tensor([])
    for i in range(feature_num - 1): # the last dimension is result
        # compute the feature_i's dimension after one-hot
        output_X_i_shape = data_num, int(input_data_copy[i].max().item())+1
        output_X_i = torch.zeros(output_X_i_shape, dtype=torch.float).scatter_(1, input_data_copy[i].view(-1,1), 1)
        output_X = torch.cat((output_X, output_X_i), 1) # put the two matrix together
    # get the label of each data
    output_y = input_data_copy[-1].view(-1, 1) * 2 - 1
    return output_X, output_y.float()

# test one_hot function
train_X, train_y = one_hot(train_dataset)
print(train_X.shape) # 17×(3+3+3+3+3+2)
print(train_y.shape) # 17×1

# add density and sugar content
new_feature = torch.tensor([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318],
               [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
               [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
               [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042],
               [0.719, 0.103]
])
train_X = torch.cat((train_X, new_feature), 1)
print(train_X.shape)

def Loss(y_hat, y):
    tmp = y * y_hat
    l = (tmp < 0).float() * tmp
    return abs(l).sum()
def createModel(input_channel, output_channel):
    net = nn.Sequential(
        nn.Linear(input_channel, output_channel)
    )
    return net

class TreeNode():
    def __init__(self, model=None, predicted=-1, left=None, right=None):
        self.model = model
        self.predicted = predicted
        self.left = left
        self.right = right


def train(net, train_X, train_y, epochs, lr, print_frequence=0):
    optim = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(epochs):
        optim.zero_grad()
        y_hat = net(train_X)
        l = Loss(train_y, y_hat)
        l.backward()
        optim.step()
        if print_frequence:
            if (epoch + 1) % print_frequence == 0:
                print("epoch:%d, loss:%f" % (epoch, l.item()))
                print("epoch:%d, accuracy:%0.2f%%\n" % (epoch, evaluate(net, train_X, train_y)))


def evaluate(net, train_X, train_y):
    y_hat = net(train_X)
    y_hat = (y_hat >= 0).float() * 2 - 1
    accuray = 100 * (y_hat == train_y).sum().float() / len(train_y)
    return accuray

def createTree(tree, train_X, train_y, epochs, lr, precision):
    if len(train_y) == 0:
        return None
    tree.model = createModel(train_X.shape[1], train_y.shape[1])
    train(tree.model, train_X, train_y, epochs, lr)
    # binnary training set according to predicted value
    train_set = binaryTrainSet(tree.model, train_X, train_y)
    # create left subtree
    if len(train_set[0][1]) == 0 or evaluate(tree.model, train_set[0][0], train_set[0][1]) > precision:
        tree.left = TreeNode(predicted=0)
    else:
        tree.left = TreeNode()
        createTree(tree.left, train_set[0][0], train_set[0][1], epochs, lr, precision)

    # create right subtree
    if len(train_set[1][1]) == 0 or evaluate(tree.model, train_set[1][0], train_set[1][1]) > precision:
        tree.right = TreeNode(predicted=1)
    else:
        tree.right = TreeNode()
        createTree(tree.right, train_set[1][0], train_set[1][1], epochs, lr, precision)

def binaryTrainSet(net, train_X, train_y):
    y_hat = net(train_X)
    train_set = [[torch.tensor([]), []] for _ in range(2)]  # create a empty list to store result
    for index in range(len(train_y)):
        class_id = int(y_hat[index] >= 0)
        train_set[class_id][0] = torch.cat((train_set[class_id][0], train_X[index].view(1, -1)), 0)
        train_set[class_id][1].append(train_y[index].item())
    for i in range(2):
        train_set[i][1] = torch.tensor(train_set[i][1], dtype=torch.float).view(-1, 1)
    return train_set


tree = TreeNode()
createTree(tree, train_X, train_y, 200, 0.01, 90)














