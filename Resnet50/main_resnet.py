import preprocessing_resnet
import train_and_test_resnet as tnt
import model_resnet
import torch



def main():
    # Will be cleaned up
    model_name = 'resnet'
    num_classes = 200
    save_path = './Resnet50/trained_models'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Create datatloaders
    dataloaders, train_set, valid_set, test_set = preprocessing_resnet.main()
    # Initialize the model 
    ResNet, optimizer, criterion = model_resnet.initialize_model(model_name=model_name, num_classes=num_classes, feature_extract = False)
    
    # Train the model
    trained_model, history = tnt.train_model(ResNet, criterion, optimizer, device, dataloaders, train_set, valid_set, num_epochs = 5)

    # Plot losses
    tnt.plot_losses(history=history)

    # Save the trained model
    tnt.save_model(trained_model=trained_model, save_path=save_path)

    # Load a trained model
    loaded_model = model_resnet.load_model(device=device, path=save_path + 'weights_test.h5', num_classes=num_classes)

    # Test a trained model
    tnt.test(loaded_model=loaded_model, device=device, dataloaders=dataloaders, test_set=test_set)

    


if __name__ == '__main__':
    main()
    