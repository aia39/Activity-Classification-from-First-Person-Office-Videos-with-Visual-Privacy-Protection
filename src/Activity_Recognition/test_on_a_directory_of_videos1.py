import torch
import sys
import numpy as np
import itertools
from model_1 import *
from model_2 import *
from model_3 import *
from model_4 import *
from final_submission_dataset import *
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
from final_package_functions import *
import os
import shutil

#testing models and generating predictions
def test_model2():
    y_pred=np.array([])
    video_name = []
    model2.eval()
    for batch_i, (X, y) in enumerate(test_dataloader1):
        with torch.no_grad():
            torch.cuda.empty_cache()
            X = X.half()
            torch.cuda.empty_cache()
            image_sequences = Variable(X.to(device), requires_grad=False)
            torch.cuda.empty_cache()
            #image_sequences = image_sequences.half()
        video_name.append(y)
        with torch.no_grad():
            # Reset LSTM hidden state
            model2.lstm.reset_hidden_state()
            # Get sequence predictions
            predictions = model2(image_sequences)
            y_pred = np.append(y_pred, predictions.detach().argmax(1).cpu().numpy())
        del(image_sequences)
        torch.cuda.empty_cache()
    return(y_pred, video_name)





if __name__ == "__main__":
    #getting the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./test_directory_folder-frames", help="Path to FPVO dataset")
    parser.add_argument("--sequence_length", type=int, default=40, help="Number of frames used in each video")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    opt = parser.parse_args()
    print(opt)

    extract_videos_frames()

    #assigning device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("\n\n\nNow our 8 model will start predicting one by one\nThe whole process involves dataloading and predicting , this might take a while \nThanks you for waiting ...")

    #creating dataset and dataloader for model 1
    test_dataset1 = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=(3, 248, 248),
        sequence_length=opt.sequence_length,
    )
    test_dataloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=1)


    

    # Define network
    model2 = ConvLSTM(
        num_classes=18,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )




    #loading weights for model 1
    model2.load_state_dict(torch.load("model_checkpoints/final_sacred_models/ConvLSTM6_9for_resnext_with_image_size_248_best_75.92(full_data).pth.pth"))

    model2.half()
    for layer in model2.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    model2 = model2.to(device)

    print('asdlfk;sdafkhsakldfjhsakldfjhafkldjhasdklfjshafkdljhsadklfjhaskfldjhakldfjhaskldfjhaskldfjhakldfjshakldfjhaskldfjhaksjdfh')
    #generating predictions for model 1
    y_pred1, video_name1 = test_model2()

    #deleting variable to save up memory
    del(model2)
    del(test_dataset1)
    del(test_dataloader1)
    torch.cuda.empty_cache()
    print("\nModel 1 is done predicting\n")


    #creating dataset and dataloader for model 2
    test_dataset1 = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=(3, 512, 512),
        sequence_length=opt.sequence_length,
    )
    test_dataloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=1)

    # Define network
    model2 = ConvLSTM2(
        num_classes=18,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )

    #model2 = model2.to(device)




    #loading weights for model 1
    model2.load_state_dict(torch.load("model_checkpoints/final_sacred_models/ConvLSTM2_0for_densenet161_with_image_size_512_batch_size_5_best_74.87(full_data).pth"))


    model2.half()
    for layer in model2.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    model2 = model2.to(device)

    print('asdlfk;sdafkhsakldfjhsakldfjhafkldjhasdklfjshafkdljhsadklfjhaskfldjhakldfjhaskldfjhaskldfjhakldfjshakldfjhaskldfjhaksjdfh')



    #generating predictions for model 2
    y_pred2, video_name2 = test_model2()

    del(model2)
    del(test_dataset1)
    del(test_dataloader1)
    torch.cuda.empty_cache()


    print("\nModel 2 is done predicting\n")

    test_dataset1 = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=(3, 324, 324),
        sequence_length=opt.sequence_length,
    )
    test_dataloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=1)


    model2 = ConvLSTM3(
        num_classes=18,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
        sequence_length=opt.sequence_length,
    )

    #model2 = model2.to(device)

    # Add weights from checkpoint model if specified
    
    


    model2.load_state_dict(torch.load("model_checkpoints/final_sacred_models/ConvLSTM5_13for_wide_resnet101_with_added_attention_image_size_324_best_75.39(full_data).pth"))


    model2.half()
    for layer in model2.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    model2 = model2.to(device)

    print('asdlfk;sdafkhsakldfjhsakldfjhafkldjhasdklfjshafkdljhsadklfjhaskfldjhakldfjhaskldfjhaskldfjhakldfjshakldfjhaskldfjhaksjdfh')
    #generating predictions for model


    y_pred3, video_name3 = test_model2()

    del(model2)
    del(test_dataset1)
    del(test_dataloader1)
    torch.cuda.empty_cache()

    print("\nModel 3 is done predicting\n")

    test_dataset1 = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=(3, 322, 322),
        sequence_length=opt.sequence_length,
    )
    test_dataloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=1)

    model2 = ConvLSTM4(
        num_classes=18,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,

    )

    #model2 = model2.to(device)

    # Add weights from checkpoint model if specified
    
    


    model2.load_state_dict(torch.load("model_checkpoints/final_sacred_models/ConvLSTM_8for_wide_resnet101_image_size_324_best_79.84(full_data).pth"))

    model2.half()
    for layer in model2.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    model2 = model2.to(device)

    print('asdlfk;sdafkhsakldfjhsakldfjhafkldjhasdklfjshafkdljhsadklfjhaskfldjhakldfjhaskldfjhaskldfjhakldfjshakldfjhaskldfjhaksjdfh')
    #generating predictions for model 1
    #y_pred1, video_name1 = test_model2()

    
    y_pred4, video_name4 = test_model2()
    del(model2)
    del(test_dataset1)
    del(test_dataloader1)
    torch.cuda.empty_cache()

    print("\nModel 4 is done predicting\n")

    
    y_pred_ensemble_1 = prediction_return(y_pred1.astype(int), y_pred2.astype(int),y_pred3.astype(int),y_pred4.astype(int))


    test_dataset1 = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=(3, 248, 248),
        sequence_length=opt.sequence_length,
    )
    test_dataloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=1)

    model2 = ConvLSTM(
        num_classes=18,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )

    #model2 = model2.to(device)

    # Add weights from checkpoint model if specified
    
    


    model2.load_state_dict(torch.load("model_checkpoints/final_sacred_model_blur/ConvLSTM_6for__resnext_with_image_size_248_best_69.73(full_data).pth"))


    model2.half()
    for layer in model2.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    model2 = model2.to(device)

    print('asdlfk;sdafkhsakldfjhsakldfjhafkldjhasdklfjshafkdljhsadklfjhaskfldjhakldfjhaskldfjhaskldfjhakldfjshakldfjhaskldfjhaksjdfh')
    #generating predictions for model 1

    y_pred1, video_name5 = test_model2()

    del (model2)
    del (test_dataset1)
    del (test_dataloader1)
    torch.cuda.empty_cache()

    test_dataset1 = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=(3, 512, 512),
        sequence_length=opt.sequence_length,
    )
    test_dataloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=1)

    model2 = ConvLSTM2(
        num_classes=18,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )

    #model2 = model2.to(device)

    # Add weights from checkpoint model if specified
    
    


    model2.load_state_dict(torch.load("model_checkpoints/final_sacred_model_blur/ConvLSTM2_0for_densenet161_with_image_size_512_batch_size_5_best_68.15(full_data).pth"))
    model2.half()
    for layer in model2.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    model2 = model2.to(device)

    print('asdlfk;sdafkhsakldfjhsakldfjhafkldjhasdklfjshafkdljhsadklfjhaskfldjhakldfjhaskldfjhaskldfjhakldfjshakldfjhaskldfjhaksjdfh')
    #generating predictions for model 1
    #y_pred1, video_name1 = test_model2()

    y_pred2, video_name6 = test_model2()
    del(model2)
    del(test_dataset1)
    del(test_dataloader1)

    torch.cuda.empty_cache()

    
    
    print("\nModel 6 is done predicting\n")



    test_dataset1 = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=(3, 324, 324),
        sequence_length=opt.sequence_length,
    )
    test_dataloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=1)

    model2 = ConvLSTM3(
        num_classes=18,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
        sequence_length=opt.sequence_length,
    )

    #model2 = model2.to(device)

    # Add weights from checkpoint model if specified


    model2.load_state_dict(torch.load("model_checkpoints/final_sacred_model_blur/ConvLSTM3_5for_wide_resnet101_with_added_attention_image_size_324_best_67.10(full_data).pth"))

    model2.half()
    for layer in model2.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    model2 = model2.to(device)

    print('asdlfk;sdafkhsakldfjhsakldfjhafkldjhasdklfjshafkdljhsadklfjhaskfldjhakldfjhaskldfjhaskldfjhakldfjshakldfjhaskldfjhaksjdfh')
    #generating predictions for model 1
    #y_pred1, video_name1 = test_model2()

    y_pred3, video_name7 = test_model2()
    del(model2)
    del(test_dataset1)
    del(test_dataloader1)
    torch.cuda.empty_cache()

    print("\nModel 7 is done predicting\n")

    test_dataset1 = Dataset(
        dataset_path=opt.dataset_path,
        input_shape=(3, 322, 322),
        sequence_length=opt.sequence_length,
    )
    test_dataloader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False, num_workers=1)

    model2 = ConvLSTM4(
        num_classes=18,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )

    #model2 = model2.to(device)

    # Add weights from checkpoint model if specified
    
    


    model2.load_state_dict(torch.load("model_checkpoints/final_sacred_model_blur/ConvLSTM4_7for_wide_resnet101_image_size_324_best_67.63(full_data).pth"))
    model2.half()
    for layer in model2.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    model2 = model2.to(device)

    print('asdlfk;sdafkhsakldfjhsakldfjhafkldjhasdklfjshafkdljhsadklfjhaskfldjhakldfjhaskldfjhaskldfjhakldfjshakldfjhaskldfjhaksjdfh')
    #generating predictions for model 1
    #y_pred1, video_name1 = test_model2()

    y_pred4, video_name8 = test_model2()
    
    del(model2)
    del(test_dataset1)
    del(test_dataloader1)
    torch.cuda.empty_cache()




    print("\nModel 8 is done predicting\n")

    
    y_pred_ensemble_2 = prediction_return2(y_pred1.astype(int), y_pred2.astype(int),y_pred3.astype(int),y_pred4.astype(int))

    y_pred = final_prediction_return(y_pred_ensemble_1.astype(int), y_pred_ensemble_2.astype(int))

    test_result(y_pred.astype(int), video_name8)
    shutil.rmtree(opt.dataset_path)