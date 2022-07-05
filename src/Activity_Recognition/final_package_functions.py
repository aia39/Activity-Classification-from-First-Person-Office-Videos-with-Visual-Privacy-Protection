import av
import glob
import os
import time
import tqdm
import datetime
import argparse
import torch
import sys
import numpy as np
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# Function to extract frames from a video
def extract_frames(video_path):
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()



def extract_videos_frames():
    prev_time = time.time()
    time_left = 0
    test_directory = './test_directory_folder'
    video_paths = glob.glob(os.path.join(test_directory, "*.MP4"))
    for i, video_path in enumerate(video_paths):
        sequence_name = video_path.replace("\\","/").split("/")[-1].split(".MP4")[0]
        sequence_path = os.path.join(f"{test_directory}-frames", sequence_name)

        if os.path.exists(sequence_path):
            continue

        os.makedirs(sequence_path, exist_ok=True)

        # Extract frames
        for j, frame in enumerate(
            tqdm.tqdm(
                extract_frames(video_path),
                desc=f"[{i}/{len(video_paths)}] {sequence_name} : ETA {time_left}",
            )
        ):
            frame.save(os.path.join(sequence_path, f"{j}.jpg"))

        # Determine approximate time left
        videos_left = len(video_paths) - (i + 1)
        time_left = datetime.timedelta(seconds=videos_left * (time.time() - prev_time))
        prev_time = time.time()



def prediction_return(pred1, pred2, pred3, pred4):
    final_pred = np.array([])

    for i in range(len(pred1)):
        if pred1[i]==pred2[i]==pred3[i]==pred4[i]:
            final_pred = np.append(final_pred, pred1[i])



        elif pred1[i] in [0,3,6,13,17,1]:
            if (pred2[i] in [2,15]):
                final_pred = np.append(final_pred,pred2[i])
            elif (pred3[i] in [16,7]):
                final_pred = np.append(final_pred,pred3[i])
            elif (pred4[i] in [4,10,9]):
                final_pred = np.append(final_pred,pred4[i])
            else:
                final_pred = np.append(final_pred,pred1[i])




        
        elif pred2[i] in [2,12,15,8,14]:
            if (pred1[i] in [0,17,6]):
                final_pred = np.append(final_pred,pred1[i])
            elif (pred4[i] in [4,10]):
                final_pred = np.append(final_pred,pred4[i])
            elif (pred3[i] == 7):
                final_pred = np.append(final_pred,pred3[i])
            else:
                final_pred = np.append(final_pred,pred2[i])





        elif pred3[i] in [7,5,16]:
            if (pred1[i] in [1,17]):
                final_pred = np.append(final_pred,pred1[i])
            elif (pred4[i] == 11):
                final_pred = np.append(final_pred,pred4[i])
            else:
                final_pred = np.append(final_pred,pred3[i])





        elif pred4[i] in [4,10,9,11]:
            if (pred1[i] in [0,17,6,13]):
                final_pred = np.append(final_pred,pred1[i])
            elif (pred2[i] in [14,15]):
                final_pred = np.append(final_pred,pred2[i])
            else:
                final_pred = np.append(final_pred,pred4[i])

        else:
            final_pred = np.append(final_pred,pred4[i])

    return(final_pred)

def prediction_return2(pred1, pred2, pred3, pred4):
    final_pred = np.array([])

    for i in range(len(pred1)):
        if pred1[i]==pred2[i]==pred3[i]==pred4[i]:
            final_pred = np.append(final_pred, pred1[i])



        elif pred1[i] in [0,1,6,7,13,16,17]:
            if (pred2[i] in [2,8]):
                final_pred = np.append(final_pred,pred2[i])
            elif (pred3[i] in [4,10,11]):
                final_pred = np.append(final_pred,pred3[i])
            elif (pred4[i] == 9):
                final_pred = np.append(final_pred,pred4[i])
            else:
                final_pred = np.append(final_pred,pred1[i])




        
        elif pred2[i] in [2,3,8,12,14,15]:
            if (pred1[i] in [0,6]):
                final_pred = np.append(final_pred,pred1[i])
            elif (pred3[i] == 4):
                final_pred = np.append(final_pred,pred3[i])
            else:
                final_pred = np.append(final_pred,pred2[i])





        elif pred3[i] in [4,5,10,11]:
            if (pred1[i] == 0):
                final_pred = np.append(final_pred,pred1[i])
            else:
                final_pred = np.append(final_pred,pred3[i])

        elif pred4[i] == 9:
            if (pred1[i] in [17,6,7,13]):
                final_pred = np.append(final_pred,pred1[i])
            else:
                final_pred = np.append(final_pred,pred4[i])

        else:
            final_pred = np.append(final_pred,pred1[i])

    return(final_pred)

def final_prediction_return(pred1,pred2):
    final_pred = np.array([])

    for i in range(len(pred1)):
        if pred1[i]==pred2[i]:
            final_pred = np.append(final_pred, pred1[i])

        elif (pred1[i] in [11,15,16]):
            final_pred = np.append(final_pred, pred1[i])

        elif (pred1[i] == 13):
            if (pred2[i] in [7,9]):
                final_pred = np.append(final_pred, pred2[i])
            else:
                final_pred = np.append(final_pred, pred1[i])

        elif (pred1[i] == 10):
            if (pred2[i] in [14,15]):
                final_pred = np.append(final_pred, pred2[i])
            else:
                final_pred = np.append(final_pred, pred1[i])

        elif (pred1[i] == 17):
            if (pred2[i] in [0,6,9,7]):
                final_pred = np.append(final_pred, pred2[i])
            else:
                final_pred = np.append(final_pred, pred1[i])

        elif (pred2[i] in [14,2,3,4,5,8,12,6]):
            final_pred = np.append(final_pred, pred2[i])

        elif (pred2[i] == 1):
            if (pred1[i] == 16):
                final_pred = np.append(final_pred, pred1[i])
            else:
                final_pred = np.append(final_pred, pred2[i])

        elif (pred2[i] == 7):
            if (pred1[i] == 11):
                final_pred = np.append(final_pred, pred1[i])
            else:
                final_pred = np.append(final_pred, pred2[i])

        elif (pred2[i] == 0):
            if (pred1[i] == 10):
                final_pred = np.append(final_pred, pred1[i])
            else:
                final_pred = np.append(final_pred, pred2[i])

        elif (pred2[i] == 9):
            if (pred1[i] in [17,13]):
                final_pred = np.append(final_pred, pred1[i])
            else:
                final_pred = np.append(final_pred, pred2[i])


        else:
            final_pred = np.append(final_pred, pred2[i])

    return(final_pred)




def test_result(predictions, video_names):
    """ Evaluate the model on the test set """
    
    # Preparing the model for evaluation

    predictions = list(predictions)
    video_names = list(video_names)

    if len(video_names) != len(predictions):
        raise RuntimeError("Mismatch in total videos and prediction")

    
    
    class_names = ['Chat', 'Clean', 'Drink', 'Dryer', 'Machine', 'Microwave', 'Mobile', 'Paper', 'Print', 'Read',
                   'Shake', 'Staple', 'Take', 'Typeset', 'Walk', 'Wash', 'Whiteboard', 'Write']
    result_list = []
    for i in range(len(predictions)):
        print("The action in the video file name " + '"' + str(video_names[i]) + '" is :' + str(class_names[predictions[i]]))
        result_list.append("The action in the video file name " + '"' + str(video_names[i]) + '" is :' + str(class_names[predictions[i]]))
    #Creating the a "log.txt" file in the "log" folder to check the predictions of our models at later time
    if not os.path.exists('log'):
        os.makedirs('log')

    log_file = open('log/log.txt', 'w')
    for each_result in result_list:
        print(each_result, file=log_file)
