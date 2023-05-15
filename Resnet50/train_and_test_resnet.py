import model_resnet
import torch
import preprocessing_resnet
import os
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import  models, datasets
import pandas as pd
from torchvision.models import  resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

"""
Todo:
- Add logger

"""

def train_model(
        model : resnet50, criterion : torch.nn.modules.loss.CrossEntropyLoss, optimizer : torch.optim.Adam,
        device : torch.device, dataloaders : torch.utils.data.DataLoader, 
        train_set : torch.utils.data.dataset.Subset, valid_set : torch.utils.data.dataset.Subset, num_epochs : int = 5
            ) -> tuple[resnet50, pd.DataFrame]:
    """
    Train a PyTorch model for a specified num_epochs. Returns a trained model and the history of the model.
    
    Parameters
    -----------
        model : resnet50
            Pretrained ResNet50 model 
        criterion : torch.nn.modules.loss.CrossEntropyLoss
            Loss function for learning criterion: cross entropy loss
        optimizer : torch.optim.Adam
            Adam optimizer
        device : torch.device
            Device to which the model is send
        dataloaders : Dataloader
            Dataloader used for loading the data onto the ResNet50 model
        train_set : Subset
            Training dataset 
        valid_set : Subset
            Validation dataset
        num_epochs : int
            Number of epochs used for training
    Returns
    -----------
        model : resnet50
            Trained ResNet50 model
        history : pd.DataFrame
            Dataframe containing the train and validation losses of the model

    """
    history = pd.DataFrame(
    columns = ['train_loss', 'validation_loss']
    )
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        dataset_size = 0
        for phase in ['train', 'validation']:
            if phase == 'train':
                dataset_size = len(train_set)
                model.train()
            else:
                dataset_size = len(valid_set)
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            history.loc[epoch, phase + '_loss'] = epoch_loss
            # history.loc[epoch, phase + '_accuracy'] = epoch_acc

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
            
    return model, history

def makedir(path) -> None:
    """
    Creates a new folder in the specified path if the folder with name folder_name does not exist yet.

    Parameters
    -----------
        path : str
            Destination path in which the folder will be made
    """
    try: 
        # mode of the folder (default)
        mode = 0o777
        # Create the new folder
        os.mkdir(path, mode)
        print("Directory '% s' is built!" % path)  
    except OSError as error: 
        print(error)


def save_model(trained_model : resnet50, save_path : str) -> None:
    """
    Makes a directory in the specified save_path and saves the weights of a model.

    Parameters
    -----------
        trained_model : resnet50
            Trained ResNet50 model 
        save_path : str
            Destination path in which the folder will be made

    """
    makedir(save_path)

    torch.save(trained_model.state_dict(), save_path + 'weights_test.h5')



def test(
        loaded_model : resnet50, device : torch.device, 
        dataloaders : torch.utils.data.DataLoader, test_set : datasets.ImageFolder
        ) -> None:
    """
    Tests a PyTorch model. 

    Parameters
    -----------
        loaded_model : resnet50
            Trained and loaded ResNet50 model
        device : torch.device
            Device to be utilized
        dataloaders : Dataloader
            Dataloader used for loading the data onto the model
        test_set : datasets.ImageFolder
            Test set of the data
    
    """
    loaded_model.eval()
    with torch.no_grad():
        n_correct = 0
        for i, (inputs,labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = loaded_model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            n_correct += torch.sum(predicted == labels.data)
        accuracy = n_correct/len(test_set)
        print('Test accuracy: {:.4f} '.format(accuracy))


def plot_losses(history : pd.DataFrame) -> None:
    """
    Plots the losses from history

    Parameters
    -----------
        history : pd.DataFrame
            Dataframe containing the train and validation losses of the model
    
    """
    history.plot( y= ['train_loss', 'validation_loss'], kind='line', figsize=(7,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and validation loss')
    plt.legend()
    plt.show()



def main():
    # ResNet, optimizer, criterion, device = model_resnet.initialize_model('resnet', num_classes=200, feature_extract = False)
    dataloaders, train_set, valid_set, test_set = preprocessing_resnet.main()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # trained_model, history = train_model(ResNet, criterion, optimizer, device, dataloaders, train_set, valid_set, num_epochs = 5)

    save_path = './Resnet50/trained_models'

    # save_model(trained_model=trained_model, save_path=save_path)

    loaded_model = model_resnet.load_model(device=device, path=save_path + 'weights_test.h5', num_classes=200)

    test(loaded_model=loaded_model, device=device, dataloaders=dataloaders, test_set=test_set)

    # plot_losses(history=history)



if __name__ == '__main__':
    main()