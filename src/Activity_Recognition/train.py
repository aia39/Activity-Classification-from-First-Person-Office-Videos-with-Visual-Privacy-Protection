import torch
import sys
import numpy as np
import itertools
from model_1 import *
from dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sampler import BalancedBatchSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


ACCURACY = 0


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]

    del(y_pred)
    del(y_true)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def test_model(epoch):
    """ Evaluate the model on the test set """
    print("")
    #defining numpy array to store the true labels and predicted labels
    y_true = np.array([])
    y_pred = np.array([])

    # Preparing the model for evaluation
    model.eval()
    test_metrics = {"loss": [], "acc": []}
    for batch_i, (X, y) in enumerate(test_dataloader):
        image_sequences = Variable(X.to(device), requires_grad=False)
        image_sequences = image_sequences.half()
        labels = Variable(y, requires_grad=False).to(device)
        y_true = np.append(y_true, labels.cpu().numpy())
        with torch.no_grad():
            # Reset LSTM hidden state
            model.lstm.reset_hidden_state()

            

            # Get sequence predictions
            predictions = model(image_sequences)

            y_pred = np.append(y_pred, predictions.detach().argmax(1).cpu().numpy())
        # Compute metrics
        acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
        loss = cls_criterion(predictions, labels).item()
        # Keep track of loss and accuracy
        test_metrics["loss"].append(loss)
        test_metrics["acc"].append(acc)
        # Log test performance
        sys.stdout.write(
            "Testing -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
            % (
                batch_i,
                len(test_dataloader),
                loss,
                np.mean(test_metrics["loss"]),
                acc,
                np.mean(test_metrics["acc"]),
            )
        )
        del(X)
        del(y)
        del(image_sequences)
        del(labels)
        del(predictions)
        del(loss)

    final_acc=np.mean(test_metrics["acc"])

    global ACCURACY

    # Save model checkpoint
    if ACCURACY < final_acc:    
        ACCURACY = final_acc
        os.makedirs("model_checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"model_checkpoints/{model.__class__.__name__}_{epoch}for_wide_resnext_image_size_248_best_{final_acc}(full_data).pth")
    # Using the learning rate scheduler while monitoring Loss and printing the learning rate after every epoch
    scheduler.step(np.mean(test_metrics["loss"]))
    for param_group in optimizer.param_groups:
        print("\nCurrent Learning Rate is : " + str(param_group['lr']))
    model.train()
    print("")
    # Getting the P, R and F score for evaluation and plotting the confusion matrix and saving that matrix
    p_score = precision_score(y_true.astype(int), y_pred.astype(int), average='macro')
    r_score = recall_score(y_true.astype(int), y_pred.astype(int), average='macro')
    f_score = f1_score(y_true.astype(int), y_pred.astype(int), average='macro')


    p_score = "Precision Score: " + str(p_score) + "\n\n"
    r_score = "Recall Score: " + str(r_score) + "\n\n"
    f_score = "F Score: " + str(f_score) + "\n\n"

    plot_title = p_score + r_score + f_score + "Confusion matrix, Without Normalization\n"

    print(y_true.astype(int))
    print(y_pred.astype(int))

    
    class_names = ['Chat', 'Clean', 'Drink', 'Dryer', 'Machine', 'Microwave', 'Mobile', 'Paper', 'Print', 'Read',
                   'Shake', 'Staple', 'Take', 'Typeset', 'Walk', 'Wash', 'Whiteboard', 'Write']
    class_names = np.array(class_names)
    plot_confusion_matrix(y_true.astype(int), y_pred.astype(int), classes=class_names, title=plot_title)

    plt.savefig(f"score{epoch}with_accuracy{final_acc}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/FPVO-segmented_videos1-frames", help="Path to FPVO dataset")
    parser.add_argument("--split_path", type=str, default="data/viptrainlist", help="Path to train/test split")
    parser.add_argument("--split_number", type=int, default=1, help="train/test split number. One of {1, 2, 3}") # this is specified when we used multiple set of train and test txt files
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=18, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=40, help="Number of frames used in each video")
    parser.add_argument("--img_dim", type=int, default=248, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument(
        "--checkpoint_interval", type=int, default=1, help="Interval between saving model checkpoints"
    )
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_shape = (opt.channels, opt.img_dim, opt.img_dim)



    # Define training set
    train_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=True,
    )
    train_dataloader = DataLoader(train_dataset, sampler=BalancedBatchSampler(train_dataset), batch_size= 18, shuffle=False, num_workers=4)# this is used to batched the data, shuffle the data and Load the data in parallel using multiprocessing workers(I don't know what that is).

    # Define test set
    test_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # Classification criterion
    cls_criterion = nn.CrossEntropyLoss().to(device)
    # Define network
    model = ConvLSTM(
        num_classes=18,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )

    #Paralleling the model to use multiple GPU
    model = nn.DataParallel(model)

    model = model.to(device)


    # convert to half precision
    model.half()  
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    # Add weights from checkpoint model if specified
    if opt.checkpoint_model:
        model.module.load_state_dict(torch.load(opt.checkpoint_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Using the scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min',threshold=0.00001,patience=4)

    #Stating the epoch
    for epoch in range(opt.num_epochs):
        epoch_metrics = {"loss": [], "acc": []}
        prev_time = time.time()
        print(f"--- Epoch {epoch}---")
        for batch_i, (X, y) in enumerate(train_dataloader):
            if X.size(0) == 1:
                continue


            image_sequences = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)

            image_sequences = image_sequences.half()

            optimizer.zero_grad()

            # Reset LSTM hidden state
            model.lstm.reset_hidden_state()

            
            # Get sequence predictions
            predictions = model(image_sequences)


            
            # Compute metrics
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()

            

            loss.backward()
            optimizer.step()

            # Keep track of epoch metrics
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = opt.num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                % (
                    epoch,
                    opt.num_epochs,
                    batch_i,
                    len(train_dataloader),
                    loss.item(),
                    np.mean(epoch_metrics["loss"]),
                    acc,
                    np.mean(epoch_metrics["acc"]),
                    time_left,
                )
            )
            del(X)
            del(y)
            del(image_sequences)
            del(labels)
            del(predictions)
            del(loss)


            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluate the model on the test set
        test_model(epoch)

        
        # Re-initialize the dataset for the next epoch
        train_dataloader = DataLoader(train_dataset, sampler=BalancedBatchSampler(train_dataset), batch_size=opt.batch_size, shuffle=False, num_workers=4)




        

