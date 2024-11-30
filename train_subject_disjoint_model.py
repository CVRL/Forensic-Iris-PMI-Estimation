import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from operator import le
from PIL import Image, ImageEnhance
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.optim as optim
from modules.dataset import SingleStreamCustomDataset, TwoStreamCustomDataset
from modules.networks import *


def train_model(args, model, criterion, optimizer, train_dataloader, test_dataloader, fold, device):
    print('Starting Training...')
    best_mae = float('inf')
    best_model_path = os.path.join(args.checkpoint, f"{args.arch}_{args.data_type}_fold_{fold+1}.pth")
    print(f"Saving best model to: {best_model_path}")

    for epoch in range(args.num_epochs):
        model.train()
        total_running_loss = 0.0

        # Training loop
        for data in train_dataloader:
            optimizer.zero_grad()

            if args.data_type == 'multi':
                filenames, nir_inputs, rgb_inputs, targets = data
                nir_inputs, rgb_inputs = (nir_inputs.to(device), rgb_inputs.to(device))
                outputs = model(nir_inputs, rgb_inputs)
            else:
                filenames, inputs, targets = data
                inputs = inputs.to(device)
                outputs = model(inputs)

            targets = targets.unsqueeze(1).to(device, dtype=torch.float)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_running_loss += loss.item()

        train_loss = total_running_loss / len(train_dataloader)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        total_mae = 0.0
        val_all_actual_values, val_all_predicted_values = [], []

        with torch.no_grad():
            for data in test_dataloader:
                if args.data_type == 'multi':
                    filenames, nir_inputs, rgb_inputs, targets = data
                    nir_inputs, rgb_inputs = (nir_inputs.to(device), rgb_inputs.to(device))
                    outputs = model(nir_inputs, rgb_inputs)
                else:
                    filenames, inputs, targets = data
                    inputs = inputs.to(device)
                    outputs = model(inputs)

                targets = targets.unsqueeze(1).to(device, dtype=torch.float)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                actual = targets.view(-1).cpu().numpy()
                predicted = outputs.view(-1).cpu().numpy()

                total_mae += mean_absolute_error(actual, predicted)
                val_all_actual_values.extend(actual)
                val_all_predicted_values.extend(predicted)

        val_loss = total_val_loss / len(test_dataloader)
        val_mae = total_mae / len(test_dataloader)

        print(f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} MAE: {val_mae:.4f}")

        # Save best model based on MAE
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with MAE: {best_mae:.4f}")

    print('Training completed! Best model saved at:', best_model_path)


def evaluate_model(args, model, test_dataloader, fold, device):
    # Define paths for the model and output
    best_model_path = os.path.join(args.checkpoint, f"{args.arch}_{args.data_type}_fold_{fold+1}.pth")
    output_path = os.path.join(args.checkpoint, f"{args.arch}_{args.data_type}_results_fold_{fold+1}.csv")

    # Load the best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Prepare to collect evaluation metrics
    all_file_names = []
    all_actual_values = []
    all_predicted_values = []

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        total_mse, total_rmse, total_mae = 0.0, 0.0, 0.0

        for data in test_dataloader:
            if args.data_type == 'multi':
                filenames, nir_inputs, rgb_inputs, targets = data
                nir_inputs, rgb_inputs = (nir_inputs.to(device), rgb_inputs.to(device))
                outputs = model(nir_inputs, rgb_inputs)
            else:
                filenames, inputs, targets = data
                inputs = inputs.to(device)
                outputs = model(inputs)

            targets = targets.unsqueeze(1).to(device, dtype=torch.float)
            actual = targets.cpu().numpy().flatten()
            predicted = outputs.cpu().numpy().flatten()

            total_mse += mean_squared_error(actual, predicted)
            total_rmse += np.sqrt(mean_squared_error(actual, predicted))
            total_mae += mean_absolute_error(actual, predicted)

            all_file_names.extend(filenames)
            all_actual_values.extend(actual)
            all_predicted_values.extend(predicted)

    # Calculate average metrics
    val_mse = total_mse / len(test_dataloader)
    val_rmse = total_rmse / len(test_dataloader)
    val_mae = total_mae / len(test_dataloader)

    # Log the performance metrics
    print('-------------------------------------')
    print('--------Best Model Performance-------')
    print('-------------------------------------')
    print(f'Fold: {fold+1}')
    print(f'MSE: {val_mse:.4f}')
    print(f'RMSE: {val_rmse:.4f}')
    print(f'MAE: {val_mae:.4f}')

    # Save the results to a CSV file
    results = pd.DataFrame({
        'Filename': all_file_names,
        'Actual': all_actual_values,
        'Predicted': all_predicted_values
    })
    results.to_csv(output_path, index=False)
    print(f'Results saved to {output_path}')


def main(args):
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current device: {device}')

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint, exist_ok=True)
    print(f"Checkpoint directory created: {args.checkpoint}")

    # Define transformations to apply to the images
    if args.arch.lower() == "inception":
        trainTransform = transforms.Compose([
            transforms.Resize(299),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(2.0)),
            transforms.ToTensor()
        ])

        testTransform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor()
        ])
    else:
        trainTransform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(2.0)),
            transforms.ToTensor()
        ])

        testTransform = transforms.Compose([
            transforms.ToTensor()
        ])

    print(f'Train transform: {trainTransform}')
    print(f'Test transform: {testTransform}')

    # Initialize the model based on architecture and data type
    if args.data_type.lower() == 'nir':
        input_channels = 1
        root_dir = args.nir_image_root_dir
        multi_stream = False
        print(f'Loading {args.data_type.lower()} data from {root_dir}')
    elif args.data_type.lower()== 'rgb':
        input_channels = 3
        root_dir = args.rgb_image_root_dir
        multi_stream = False
        print(f'Loading {args.data_type.lower()} data from {root_dir}')
    elif args.data_type.lower() == 'multi':
        nir_dir = args.nir_image_root_dir
        rgb_dir = args.rgb_image_root_dir
        multi_stream = True
        print(f'Loading {args.data_type.lower()} data from {nir_dir} <> {rgb_dir}')
    else:
        raise ValueError(f"Unsupported data type: {args.data_type}")

    print(f'Data type: {args.data_type}')
    print(f'Pretrained weights:{args.pretrained}')

    # Define the model based on the architecture and data type
    if args.data_type != 'multi':  # Single stream models
        if args.arch == 'alexnet':
            model = CustomAlexNet(input_channels=input_channels, pretrained=args.pretrained, num_classes=1)
        elif args.arch == 'vgg':
            model = CustomVGG19(input_channels=input_channels, pretrained=args.pretrained, num_classes=1)
        elif args.arch == 'resnet':
            model = CustomResNet152(input_channels=input_channels, pretrained=args.pretrained, num_classes=1)
        elif args.arch == 'inception':
            model = CustomInception(input_channels=input_channels, pretrained=args.pretrained, num_classes=1)
        elif args.arch == 'densenet':
            model = CustomDenseNet121(input_channels=input_channels, pretrained=args.pretrained, num_classes=1)
        elif args.arch == 'vit':
            model = CustomViT(input_channels=input_channels, num_classes=1)
        else:
            raise ValueError(f"Unsupported model architecture: {args.arch} for single stream data type")
    else:  # Two-stream models for multispectral data
        if args.arch == 'alexnet':
            model = TwoStreamAlexNet(pretrained=args.pretrained, num_classes=1)
        elif args.arch == 'vgg':
            model = TwoStreamVGG(pretrained=args.pretrained, num_classes=1)
        elif args.arch == 'resnet':
            model = TwoStreamResNet(pretrained=args.pretrained, num_classes=1)
        elif args.arch == 'inception':
            model = TwoStreamInception(pretrained=args.pretrained, num_classes=1)
        elif args.arch == 'densenet':
            model = TwoStreamDenseNet(pretrained=args.pretrained, num_classes=1)
        elif args.arch == 'vit':
            model = TwoStreamViT(num_classes=1)
        else:
            raise ValueError(f"Unsupported model architecture: {args.arch} for multispectral data type")

    # Move the model to the device
    model = model.to(device)
    print(model)

    # Wrap model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for model parallelism.")
        model = nn.DataParallel(model)
        torch.backends.cudnn.enabled = True

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    if args.solver_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6 if args.weight_decay else 0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6 if args.weight_decay else 0)

    print(f'Weight decay:{args.weight_decay}')
    print(f'Loss function:{criterion}')
    print(f'Optimizer:{optimizer}')

    # Loop through each fold
    for fold in range(0, 10):
        print(f"Starting fold {fold + 1}")

        # Load data
        train_data = pd.read_csv(f"{args.metadata_file_path}fold_{fold+1}_train.csv")
        val_data = pd.read_csv(f"{args.metadata_file_path}fold_{fold+1}_validation.csv")
        test_data = pd.read_csv(f"{args.metadata_file_path}fold_{fold+1}_test.csv")

        if multi_stream:
            train_dataset = TwoStreamCustomDataset(train_data, nir_dir, rgb_dir, transform=trainTransform)
            val_dataset = TwoStreamCustomDataset(val_data, nir_dir, rgb_dir, transform=testTransform)
            test_dataset = TwoStreamCustomDataset(test_data, nir_dir, rgb_dir, transform=testTransform)
        else:
            train_dataset = SingleStreamCustomDataset(train_data, root_dir, transform=trainTransform)
            val_dataset = SingleStreamCustomDataset(val_data, root_dir, transform=testTransform)
            test_dataset = SingleStreamCustomDataset(test_data, root_dir, transform=testTransform)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print(f'Trainset length: {len(train_loader)}')
        print(f'Validation length: {len(val_loader)}')
        print(f'Testset length: {len(test_loader)}')
        
        # Training and evaluating model
        train_model(args, model, criterion, optimizer, train_loader, val_loader, fold, device)
        evaluate_model(args, model, test_loader, fold, device)

        print(f"Completed fold {fold + 1}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iris Image-based Post-mortem Interval Estimation')

    # Image Directories
    parser.add_argument('--nir_image_root_dir', type=str, default='./pm-iris-dataset/NIR/', help='Root directory for NIR images')
    parser.add_argument('--rgb_image_root_dir', type=str, default='./pm-iris-dataset/RGB/', help='Root directory for RGB images')

    # Metadata Path
    parser.add_argument('--metadata_file_path', type=str, default='./train-testset/sub-disj/NIR/', help='Path to metadata')

    # Data Type
    parser.add_argument('--data_type', type=str, default='nir', help='Type of data being used')

    # Model Parameters
    parser.add_argument('--arch', type=str, default='vgg', help='Model architecture to use (e.g., vgg, resnet, etc.)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model weights')
    parser.add_argument('--solver_name', type=str, default='Adam', help='Optimizer to use (e.g., Adam, SGD)')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', action='store_true', help='Weight decay for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train the model')
    parser.add_argument('--checkpoint', type=str, default='./models-checkpoint/sub-disj/NIR/', help='Directory to save the model checkpoint')

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)


