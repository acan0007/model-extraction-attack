import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from attackModel import lenet, lenet_a
from sklearn.metrics import classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
#import random
#import numpy as np
#import torch.nn as nn

def main():
    # Enable training with GPU, if not available use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start timing
    start_time = time.time()

    # Data transformation and loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('./data', download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

    # Load target model
    target_model = lenet().to(device)
    target_model.load_state_dict(torch.load('mnist_cnn.pt'))
    target_model.eval()

    # Initialize attack model
    attack_model = lenet_a().to(device)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)
    criterion = torch.nn.NLLLoss()

    # Calculate mean and covariance of training data
    def calculate_mean_covariance(loader):
        data_list = []
        for data, _ in loader:
            data_list.append(data.view(data.size(0), -1))
        data_tensor = torch.cat(data_list)
        mean = torch.mean(data_tensor, dim=0)
        cov = torch.cov(data_tensor.T)
        return mean, cov

    mean, cov = calculate_mean_covariance(loader)
    
    epsilon = 1e-5 
    cov+= torch.eye(cov.size(0)) * epsilon
    cov_inv = torch.linalg.inv(cov)

    # Mahalanobis distance for OOD detection
    def mahalanobis_distance(data, mean, cov_inv):
        data = data.view(data.size(0), -1)
        diff = data - mean
        md = torch.sqrt(torch.diag(torch.mm(torch.mm(diff, cov_inv), diff.T)))
        return md
    
    def is_ood(data):
        md = mahalanobis_distance(data, mean, cov_inv)
        md_threshold = 90  # Adjust this threshold based on the results/observations
        prob_threshold = 0.55 # Adjust this threshold for OOD detection
        
        # Model to get output logits for probability-based OOD detection
        logits = target_model(data)
        probs = F.softmax(logits, dim=1)
        max_prob, _ = torch.max(probs, dim=1)
        
        is_md_ood = (md > md_threshold).any().item()
        is_prob_ood = (max_prob < prob_threshold).any().item()
        # Print debugging information
        #print(f"Mahalanobis Distance: {md}, Threshold: {md_threshold},is_md_ood: {is_md_ood}, is_prob_ood: {is_prob_ood}")
        return is_md_ood or is_prob_ood

    # Attack training loop with defense mechanism
    for epoch in range(5):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = target_model(data)
            predicted_labels = output.max(1)[1]

            # Defense: Modify response to add noise for OOD queries.
            if is_ood(data):
                # Varying noise dynamically for OOD queries
                noise = torch.rand_like(output) * torch.FloatTensor(1).uniform_(128  , 512)
                output += noise
                predicted_labels = output.max(1)[1]
                #print("string resp is OOD") # debug print of OOD queries
            else:
                predicted_labels = output.max(1)[1]
                #print("string resp is ID") # debug print of the ID queries

            # Train attack model
            optimizer.zero_grad()
            attack_output = attack_model(data)
            loss = criterion(attack_output, predicted_labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Save the attack model
    torch.save(attack_model.state_dict(), 'attack_model_defended.pth')

    # Evaluate the attack model
    attack_model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = attack_model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Print total accuracy of the attack model
    accuracy = 100 * correct / total
    print(f'Accuracy of the attack model on the test images: {accuracy}%')

    # Print classification report and confusion matrix
    report = classification_report(all_labels, all_preds)
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('conf_bias_defence.png')

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time taken for attack: {total_time:.2f} seconds')

if __name__ == "__main__":
    main()