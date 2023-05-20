import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision.models import  resnet50, ResNet50_Weights

def initialize_model(model_name : str, num_classes : int, 
                     feature_extract : bool
                     ) -> tuple[resnet50, torch.optim.Adam, torch.nn.modules.loss.CrossEntropyLoss]:
    """
    Initializes a pretrained model, adds a linear layer for predicting the specified num_classes.
    Also sends the model to the device.
    Defines the learning optimizer and criterion. Returns the initialized model, criterion, and optimizer.

    Parameters
    -----------
        model_name : str
            Name of the model that needs to be initialized
        num_classes : int
            Number of classes the model needs to be able to classify
        feature_extract : bool
            Boolean stating if the base of the model needs to be frozen

    Returns
    -----------
        model : resnet50
            ResNet50 initialized model with an added linear layer for predicting the specified num_classes
        optimizer : torch.optim.Adam
            Adam optimizer
        criterion : torch.nn.modules.loss.CrossEntropyLoss
            Loss function for learning criterion: cross entropy loss
    """
    
    model = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if model_name == 'resnet':
        """
        ResNet50 model
        """
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        
        for param in model.parameters():
            param.requires_grad = feature_extract   
        
        # Add last linear layer for prediction of num_classes
        model.fc = nn.Sequential(
                    # nn.Linear(2048, 128),
                    # nn.ReLU(inplace=True),
                    nn.Linear(2048, num_classes)).to(device)


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters())

    return model, optimizer, criterion

def load_model(device : torch.device, path : str, num_classes : int) -> resnet50:
    """
    Loads a trained model and sends it to the device. Returns the loaded model.

    Parameters
    -----------
        device : torch.device
            Device to which the model is send
        path : str
            Destination path of where the model weights are stored
        num_classes : int
            Number of classes the model needs to be able to classify
    """
    loaded_model = resnet50(weights =ResNet50_Weights.DEFAULT).to(device)
    loaded_model.fc = nn.Sequential(
                # nn.Linear(2048, 128),
                # nn.ReLU(inplace=True),
                nn.Linear(2048, num_classes)).to(device)
    
    loaded_model.load_state_dict(torch.load(path))
    return loaded_model


def main():
    model, optimizer, criterion = initialize_model('resnet', num_classes=200, feature_extract = False)

    return model, optimizer, criterion


if __name__ == '__main__':
    main()