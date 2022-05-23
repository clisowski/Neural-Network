import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    if (training == True):
        train_set = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]))
        loader = torch.utils.data.DataLoader(train_set, batch_size = 50)
    else:
        test_set = datasets.MNIST('./data', train=False,
                       transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]))
        loader = torch.utils.data.DataLoader(test_set, batch_size = 50)
    
    return loader

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128, bias = True),
                         nn.ReLU(), nn.Linear(128,64, bias = True),
                         nn.ReLU(), nn.Linear(64, 10, bias = True))
    return model

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    model.train()
    avg = 0
    
    for epoch in range(0, T):
        running_loss = 0.0
        correct = 0
        total=0
        avg = 0
        for batch_idx, (data_, target_) in enumerate(train_loader):
            opt.zero_grad()
            outputs = model(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            avg += loss.item() * 50
        print ('Train Epoch: {}  Accuracy: {}/{}({:.2f}%)  Loss: {:.3f}' 
                    .format(epoch, correct, total, 100 * correct / total, avg / len(train_loader.dataset)))
        
    return None

def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    iteration = 0
    avg = 0
    with torch.no_grad():
        correct = 0
        total=0
        for batch_idx,(data,target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output,target)
            _,pred_t = torch.max(output, dim=1)
            correct += torch.sum(pred_t==target).item()
            total += target.size(0)
            avg += loss.item()
        if show_loss:
            print("Average loss: {:.4f} \nAccuracy: {}%".format(avg / len(test_loader.dataset), 100 * correct / total))
        else:
            print("Accuracy: {}%".format(100 * correct / total))


    return None
    
def predict_label(model, test_images, index):
    #model - trained model
    #test_images - test image set of shape Nx1x28x28
    #index - specific index i of the image to be tested: 0<=i<=N-1
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    image = test_images[index]
    output = model(image)
    prob = F.softmax(output, dim=1)
    sorted1 = torch.sort(prob, descending = True)
    for i in range(3):
        print("{}: {:.2f}%".format(class_names[sorted1[1][0][i].item()], sorted1[0][0][i].item() * 100))
    return None
    

            



if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
