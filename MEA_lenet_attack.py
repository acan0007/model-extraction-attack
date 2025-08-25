import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#target model used
from attackModel import  lenet, lenet_a
import time
import numpy as np

def main():
    # Enable train with gpu, if not available use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # start timer for timing/cost computation
    start_time = time.time()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load and prepare data
    dataset = datasets.MNIST('./data', download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)

    # Load target model
    target_model = lenet()
    target_model.load_state_dict(torch.load('mnist_cnn.pt'))
    target_model.eval()

    # Initialize attack model
    attack_model = lenet_a()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)
    criterion = torch.nn.NLLLoss()

    for epoch in range(50):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = target_model(data)
            predicted_labels = output.max(1)[1]

            # Train attack model
            optimizer.zero_grad()
            attack_output = attack_model(data)
            loss = criterion(attack_output, predicted_labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        #if epoch % 100 == 0:
           #print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Save the attack model
    torch.save(attack_model.state_dict(), 'attack_model.pth')

    # Evaluate the attack model
    attack_model.eval()
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            output = attack_model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_labels.extend(target.cpu().numpy())
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    #confussion matrix (bias)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_predictions)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot(cmap=plt.cm.Blues)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('conf_bias.png')
    #plt.show()

    # classification report (f1 score; macro, weighted average)
    from sklearn.metrics import classification_report
    report = classification_report(all_targets, all_predictions, digits=4)
    print(report)

    # print total accuracy of the attack model
    accuracy = 100 * np.mean(np.array(all_predictions) == np.array(all_targets))
    print(f'Accuracy of the attack model on the test images: {accuracy}%')

    # end timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time taken for attack: {total_time:.2f} seconds')

if __name__ == "__main__":

    main()
