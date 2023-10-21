import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import model
import ResNet


def get_CIFAR10_data():
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [45000, 5000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return dataset, train_set, val_set, test_set, train_loader, val_loader, test_loader, classes


def model_train():
    loss_hist, acc_hist = [], []
    loss_hist_val, acc_hist_val = [], []

    for epoch in range(epochs):
        time_ckpt = time.time()
        print("EPOCH:", epoch + 1, end=" ")
        running_loss = 0.0
        correct = 0
        for data in train_loader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute training statistics
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_set)
        avg_acc = correct / len(train_set)
        loss_hist.append(avg_loss)
        acc_hist.append(avg_acc)

        # validation statistics
        model.eval()
        with torch.no_grad():
            loss_val = 0.0
            correct_val = 0
            for data in val_loader:
                batch, labels = data
                batch, labels = batch.to(device), labels.to(device)
                outputs = model(batch)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                loss_val += loss.item()

            avg_loss_val = loss_val / len(val_set)
            avg_acc_val = correct_val / len(val_set)
            loss_hist_val.append(avg_loss_val)
            acc_hist_val.append(avg_acc_val)
        model.train()
        scheduler.step(avg_loss_val)

        print("Training Loss: {:.3f}".format(avg_loss * 100), end=" ")
        print("Val Loss: {:.3f}".format(avg_loss_val * 100), end=" ")
        print("Train Accuracy: {:.2f}%".format(avg_acc * 100), end=" ")
        print("Val Accuracy: {:.2f}%".format(avg_acc_val * 100), end=" ")

        with open("Models/cnn_cifar10_model_{}.pth".format(epoch + 1), "wb") as f:
            model.eval()
            pickle.dump(model, f)
            model.train()
        print("Time: {:.2f} sec".format(time.time() - time_ckpt), end=" \n")

    return loss_hist, loss_hist_val, acc_hist, acc_hist_val


def complementary_show(show: bool):
    if show:
        # create confusion matrix
        confusion_mat = confusion_matrix(label_vec, pred_vec)
        labels = np.unique(label_vec)
        confusion_df = pd.DataFrame(confusion_mat, index=classes, columns=classes)
        print("Confusion Matrix")
        print(confusion_df)

        # create a report to show the f1-score, precision, recall
        report = pd.DataFrame.from_dict(classification_report(pred_vec, label_vec, output_dict=True)).T
        report['Label'] = [classes[int(x)] if x.isdigit() else " " for x in report.index]
        report = report[['Label', 'f1-score', 'precision', 'recall', 'support']]
        print(report)


if __name__ == '__main__':
    # hyperparameters
    epochs = 80
    learning_rate = 0.01
    batch_size = 128
    num_workers = 2
    # model = model.Net()
    # model = torchvision.models.resnet50(pretrained=False)
    model = ResNet.ResNet18()
    comp_show = True
    # 基础准确率：76.28%
    # 50resnet 82.66%
    # Resnet 90.66%

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # get dataset
    dataset, train_set, val_set, test_set, train_loader, val_loader, test_loader, classes = get_CIFAR10_data()

    # record parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", num_params)

    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = \
        optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = \
        optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=0)

    # train model
    loss_hist, loss_hist_val, acc_hist, acc_hist_val = model_train()

    # loss visualization
    plots = [(loss_hist, loss_hist_val), (acc_hist, acc_hist_val)]
    plt_labels = [("Training Loss", "Validation Loss"), ("Training Accuracy", "Validation Accuracy")]
    plt_titles = ["Loss", "Accuracy"]
    plt.figure(figsize=(20, 7))
    for i in range(0, 2):
        ax = plt.subplot(1, 2, i + 1)
        ax.plot(plots[i][0], label=plt_labels[i][0])
        ax.plot(plots[i][1], label=plt_labels[i][1])
        ax.set_title(plt_titles[i])
        ax.legend()
    plt.show()

    # selecting the best model and epoch
    best_acc = max(acc_hist_val)
    best_epoch = acc_hist_val.index(best_acc) + 1
    print("Best accuracy on validation set: {:.2f}%".format(best_acc * 100))
    print("Best epoch: {}".format(best_epoch))
    with open(f"Models/cnn_cifar10_model_{best_epoch}.pth", "rb") as f:
        loaded_model = pickle.load(f)

    # test the Trained Network
    pred_vec = []
    label_vec = []
    correct = 0
    test_loss = 0.0
    model.to(device)
    with torch.no_grad():
        for data in test_loader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)
            outputs = model(batch)
            test_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            pred_vec.extend(predicted.cpu().numpy())  # Convert tensor to numpy array
            label_vec.extend(labels.cpu().numpy())  # Convert tensor to numpy array
    pred_vec = np.array(pred_vec)
    label_vec = np.array(label_vec)
    print("Test Loss: {:.2f}".format(test_loss))
    print('Test Accuracy on the 10000 test images: %.2f %%' % (100 * correct / len(test_set)))

    # compute the Accuracy, F1-Score, Precision, Recall, Support
    complementary_show(comp_show)






