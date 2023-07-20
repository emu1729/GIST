import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from networks import one_layer_net, ConcatenatedNetwork
import copy


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--image_embedding_file', type=str, default=None, metavar='N',
                    help='model name')
parser.add_argument('--metadata', type=str, default=None, metavar='N',
                    help='metadata file')
parser.add_argument('--num_epochs', type=int, default=1000, metavar='N',
                    help='training epochs')
parser.add_argument('--lr', type=float, default=0.5, metavar='N',
                    help='learning rate')
parser.add_argument('--output_file', type=str, default=None, metavar='N',
                    help='where to save output')


def run_linear_probe(image_embedding_file, metadata, num_epochs, lr, output_file):
    image_embeddings = pickle.load(open(image_embedding_file, 'rb'))
    
    df = pd.read_csv(metadata)
    df = df[df['split'].isin(['train', 'val', 'test'])]
    print(df.shape)

    labels = {}
    train_keys = []
    val_keys = []
    test_keys = []
    for i, row in df.iterrows():
        labels[row['filename']] = row['label']
        if row['split'] == 'train':
            train_keys.append(row['filename'].split('/')[-1].split('.')[0])
        if row['split'] == 'val':
            val_keys.append(row['filename'].split('/')[-1].split('.')[0])
        if row['split'] == 'test':
            test_keys.append(row['filename'].split('/')[-1].split('.')[0])
    
    #image_embeddings = {key: value.flatten() for key, value in image_embeddings.items()}
    image_embeddings = {key.split('/')[-1].split('.')[0]: np.array(value) for key, value in image_embeddings.items()}
    label_embeddings = {key.split('/')[-1].split('.')[0]: np.array(value) for key, value in labels.items()}
    
    print("Creating data...")
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    
    for key in train_keys:
        X_train.append(image_embeddings[key])
        y_train.append(label_embeddings[key])
    for key in val_keys:
        X_val.append(image_embeddings[key])
        y_val.append(label_embeddings[key])
    for key in test_keys:
        X_test.append(image_embeddings[key])
        y_test.append(label_embeddings[key])

    X_train = torch.tensor(np.concatenate(X_train, axis=0), dtype=torch.float32)
    X_train = X_train.squeeze()
    #X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train))
    print(X_train.size())
    print(y_train.size())
    X_val = torch.tensor(np.array(X_val), dtype=torch.float32)
    X_val = X_val.squeeze()
    y_val = torch.tensor(np.array(y_val))
    print(X_val.size())
    print(y_val.size())
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    X_test = X_test.squeeze()
    y_test = torch.tensor(np.array(y_test))
    print(X_test.size())
    print(y_test.size())
    
    print("Setting up model...")
    # Create a TensorDataset
    dataset = TensorDataset(X_train, y_train)
    dataset_val = TensorDataset(X_val, y_val)
    dataset_test = TensorDataset(X_test, y_test)
    
    # Create a data loader for batch training
    batch_size = 64
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # Define the single-layer network
    input_size = len(X_train[0])
    if 'fitzpatrick' in dataset_name:
        output_size = 40
    elif 'cub' in dataset_name:
        output_size = 200
    elif 'flower' in dataset_name:
        output_size = 102
    elif 'aircraft' in dataset_name:
        output_size = 100
    
    print(input_size)
    print(output_size)
    if network == 'linear':
        net = torch.nn.Linear(input_size, output_size)
    
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    # Training loop
    val_accuracies = []
    best_accuracy = 0.0
    best_weights = None
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            optimizer.zero_grad()
    
            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)
    
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
    
            # Accumulate loss
            running_loss += loss.item()
    
        # Print average loss for the epoch
        avg_loss = running_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
        correct = 0
        total = 0
    
        with torch.no_grad():
            for inputs, labels in data_loader_val:
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        accuracy = correct / total
        val_accuracies.append(accuracy)
    
        if accuracy >= best_accuracy:
            print('Validation Accuracy:', accuracy)
            best_accuracy = accuracy
            best_weights = copy.deepcopy(net.state_dict())
    
    net.load_state_dict(best_weights)
    
    correct = 0
    total = 0
    
    hidden_outputs = []
    labels_all = []
    final_outputs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader_train:
            outputs = net(inputs)
            labels_all.append(labels)
            final_outputs.append(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Train Accuracy: {accuracy:.2%}')
    
    correct = 0
    total = 0
    top3_correct = 0
    
    hidden_outputs_val = []
    labels_val = []
    final_outputs_val = []
    
    with torch.no_grad():
        for inputs, labels in data_loader_val:
            outputs = net(inputs)
            labels_val.append(labels)
            final_outputs_val.append(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Val Accuracy: {accuracy:.2%}')
    
    print(max(val_accuracies))
    
    correct = 0
    total = 0
    
    hidden_outputs_test = []
    labels_test = []
    final_outputs_test = []
    
    with torch.no_grad():
        for inputs, labels in data_loader_test:
            outputs = net(inputs)
            labels_test.append(labels)
            final_outputs_test.append(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
            # Calculate top-3 accuracy
            _, top3_indices = torch.topk(outputs.data, k=3)
            top3_correct += sum(labels[i] in top3_indices[i] for i in range(labels.size(0)))
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.2%}')
    top3_accuracy = top3_correct / total
    print(f'Test Top3 Accuracy: {top3_accuracy:.2%}')
    
    d = {'train': [final_outputs, labels_all],
         'val': [final_outputs_val, labels_val],
         'test': [final_outputs_test, labels_test]}
    
    if output_file:
        pickle.dump(d, open(output_file, 'wb'))
    
    
    # Initialize variables
    num_bootstraps = 1000
    
    import numpy as np
    from sklearn.utils import resample
    
    
    labels_test = [arr.numpy() for arr in labels_test]
    labels_test = np.concatenate(labels_test, axis=0)
    final_outputs_test = [arr.numpy() for arr in final_outputs_test]
    final_outputs_test = np.concatenate(final_outputs_test, axis=0)
    
    all_top1_accs = []
    all_top3_accs = []
    
    for j in range(1000):
        print(j)
        labels, outputs = resample(labels_test, final_outputs_test, replace=True, random_state=j)
        labels = torch.from_numpy(labels).cuda()
        outputs = torch.from_numpy(outputs).cuda()
    
        _, predicted = torch.max(outputs, 1)
    
        correct_top1 = (labels == predicted).sum().item()
    
        _, top3_indices = torch.topk(outputs, k=3)
    
        correct_top3 = sum(labels[i] in top3_indices[i] for i in range(labels.size(0)))
    
        all_top1_accs.append(correct_top1 / labels.size(0))
        all_top3_accs.append(correct_top3 / labels.size(0))
    
    mean_top1_acc = np.mean(all_top1_accs)
    std_top1_acc = np.std(all_top1_accs)
    mean_top3_acc = np.mean(all_top3_accs)
    std_top3_acc = np.std(all_top3_accs)
    
    print(f'Top-1 Accuracy: Mean = {mean_top1_acc:.2%}, Std = {std_top1_acc:.2%}')
    print(f'Top-3 Accuracy: Mean = {mean_top3_acc:.2%}, Std = {std_top3_acc:.2%}')


if __name__ == '__main__':
    args = parser.parse_args()
    run_linear_probe(args.image_embedding_file, args.metadata, args.num_epochs, args.lr, args.output_file)