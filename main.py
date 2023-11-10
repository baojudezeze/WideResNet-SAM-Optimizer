import argparse

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import model
import pyramidnet
import utils


def get_CIFAR10_data(batch_size, num_workers):
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    data = torch.cat([d[0] for d in DataLoader(train_set)])

    # data augmentation
    train_transform = transforms.Compose([
        torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])),
        utils.Cutout()
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3]))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_set, val_set, train_loader, val_loader, classes


def model_train():
    for epoch in range(args.epochs):

        model.train()
        log.train(len_dataset=len(train_loader))

        # train loss
        for data in train_loader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)

            # the SAM Optimizer, which needs two forward-backward passes to estimate the "sharpness-aware" gradient
            # step 1
            utils.enable_running_stats(model)
            outputs = model(batch)
            loss = utils.smooth_crossentropy(outputs, labels, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # step 2
            utils.disable_running_stats(model)
            utils.smooth_crossentropy(model(batch), labels, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(outputs.data, 1) == labels
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(val_loader))

        # val loss
        with torch.no_grad():
            for data in val_loader:
                batch, labels = data
                batch, labels = batch.to(device), labels.to(device)
                outputs = model(batch)
                loss = utils.smooth_crossentropy(outputs, labels)
                correct = torch.argmax(outputs, 1) == labels
                log(model, loss.cpu(), correct.cpu())
    log.flush()


def complementary_show(show: bool):
    if show:
        # create confusion matrix
        confusion_mat = confusion_matrix(label_vec, pred_vec)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--depth", default=28, type=int)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.1 for label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--comp_show", default=True, type=bool)
    args = parser.parse_args()

    # cudnn settings
    utils.initialize(seed=42)

    # device and model
    device = torch.device("cuda")
    model = model.WideResNet(args.depth, args.width_factor, dropout=args.dropout, in_channels=3, labels=10).to(device)
    # model = model.PyramidNet(dataset='cifar10', depth=110, alpha=64, num_classes=10, bottleneck=False)
    # model = pyramidnet.PyramidNet(dataset='cifar10', depth=272, alpha=48, num_classes=10, bottleneck=True).cuda()

    # get dataset
    train_set, val_set, train_loader, val_loader, classes = get_CIFAR10_data(args.batch_size, args.num_workers)

    # log
    log = utils.Log(log_each=10)

    # record parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", num_params)

    # optimizer and scheduler
    optimizer = utils.SAM(model.parameters(), torch.optim.SGD, rho=args.rho, adaptive=True,
                          lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = utils.StepLR(optimizer, args.learning_rate, args.epochs)

    # train model
    model_train()

    # test the Trained Network
    device = torch.device("cuda")
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])
    pred_vec = []
    label_vec = []
    correct = 0
    test_loss = 0.0
    model.to(device)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    with torch.no_grad():
        for data in test_loader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)
            outputs = model(batch)
            test_loss = utils.smooth_crossentropy(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            pred_vec.extend(predicted.cpu().numpy())
            label_vec.extend(labels.cpu().numpy())
    pred_vec = np.array(pred_vec)
    label_vec = np.array(label_vec)
    test_loss = np.array(test_loss.cpu())
    print("Test Loss: {:.2f}".format(test_loss[-1]))
    print('Test Accuracy on the 10000 test images: %.2f %%' % (100 * correct / len(test_set)))

    # best 96.93 % wrn epoch120
    # best 97.27 % wrn epoch200
    # best 96.26% pyramid epoch200

    # compute the Accuracy, F1-Score, Precision, Recall, Support
    complementary_show(args.comp_show)
