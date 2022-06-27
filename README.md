# Activity-Classification-from-First-Person-Office-Videos-with-Visual-Privacy-Protection
This is the official repository of the paper titled 'Activity Classification from First Person Office Videos with Visual Privacy Protection' which is accepted in IC4IR 2021. This is an open source activity (18 classes) recognition from human concerned privacy protected video codebase. We provide both the training and testing scripts along with model weight for any comparison purposes. This repository includes implementations of the following methods
* Privacy protected video generation using Mask R-CNN (backbone Inception ResNet V2)
* Video classification on raw rgb frames.
* Video classification on blurred (yielded from privacy protection scheme) rgb frames.
* Video classification from mixed training (leveraging both the rgb and blurred video learning)


# Introduction
The goal of our approach is to provide a high-performing, visually privacy aware, modular codebase which provides user to make the most use of any module of our work.
In the visual privacy protection scheme Inception ResNet V2 is used as backbone and Mask-RCNN is used to detect specific masks. Firstly, inspecting the whole [FPV-O](http://www.eecs.qmul.ac.uk/~andrea/fpvo) dataset which was provided in [Video and Image Processing Cup 2019](https://signalprocessingsociety.org/community-involvement/vip-cup-2019-icip-2019) we identified privacy violating sensitive objects for each class. All the sensitive objects are listed in below table. 

Distribution of Sensitive Objects in Different Classes
| Activity Class | Sensitive Objects |
| -- | -- |
| Chat | Person, digital screen, keyboard, laptop, book |
| Clean | Person  |
| Drink | Person, Digital Screen |
| Dryer | Person, Toilet |
| Machine | Person, Digital Screen |
| Microwave | Person |
| Mobile | Mobile, Digital Screen, Keyboard, Book, Person |
| Paper | Book, Laptop, Digital Screen, Keyboard, Person | 
| Print | Person, Digital Screen, Laptop, Book | 
| Read | Digital Screen, Keyboard, Person, Laptop |
| Shake | Digital Screen, Laptop, Person |
| Staple | Book, Digital Screen, Keyboard, Person |
| Take | Digital Screen, Person |
| Typeset | Digital Screen, Keyboard, Person, Laptop |
| Walk | Digital Screen, Keyboard, Laptop |
| Wash | Toilet, Person |
| Whiteboard | Person, Book |
| Write | Digital Screen, Keyboard, Person, Laptop, Book |

In the activity recognition section, we developed an ensembled based LSTM model with attention which trained on mixed dataset (described in the paper). Ensemble on the models are:
* ResNeXt 101 + Attention
* WideResNet 101 + Attention
* DenseNet + Attention
* WideResNet 101

## Folder Directory Hierarchy (Visual Privacy Protection module) 

```
test_directory_folder/
        video1.mp4
        video2.mp4
        ...
        
```
Output will be saved in **protected_directory_folder**

## Folder Directory Hierarchy (Activity Recognition module) 

```
Parent_folder/
    Class 1/
        video1.mp4
        video2.mp4

        ...
        
    Class 2/
        video1.mp4
        video2.mp4

        ...
```       




# Installation (Modular Implementation)
For **Visual Privacy Protection module**,
1. Install dependencies 
   ```bash
   conda install -c anaconda tensorflow-gpu
   conda install -c anaconda pillow
   conda install -c anaconda opencv
   conda install -c anaconda matplotlib
   ``` 
2. Download pre-trained weights,classes names and related file from the [Tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Download 'mask_rcnn_inception_resnet_v2_atrous_coco' from 'COCO-trained models' table. Put the .rar file in ['object_detection'](https://github.com/aia39/Activity-Classification-from-First-Person-Office-Videos-with-Visual-Privacy-Protection/src/Privacy protected video generation/object_detection) folder. You can also download other models which gives mask as output. We select inception_resnet_V2 as it gives better result though computationally expensive.

3. Create two folders in 'object_detection' folder 'test_directory_folder', 'protected_directory_folder' respectively for test videos and protected videos.

4. Run the [py](https://github.com/aia39/Activity-Classification-from-First-Person-Office-Videos-with-Visual-Privacy-Protection/src/Privacy protected video generation/object_detection/protected.py) file in command window from 'object_detection' folder to generate masked video/frame.
 ```bash
   python protected.py
   ```

For **Activity Recognition module**,
1. Install dependencies 
   ```bash
   conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
   conda install -c conda-forge tqdm
   conda install -c conda-forge av
   conda install -c anaconda pillow
   conda install numpy matplotlib scikit-learn
   ``` 
 2.
 
 
 
 
## Acknowledgement 
Thanks to [mhealth lab](https://mhealth.buet.ac.bd/) for providing world class research facility. 

## Activity-Classification-from-First-Person-Office-Videos-with-Visual-Privacy-Protection
If you find 'Activity-Classification-from-First-Person-Office-Videos-with-Visual-Privacy-Protection' useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@article{ghosh2020privacy,
  title={Privacy-Aware Activity Classification from First Person Office Videos},
  author={Ghosh, Partho and Istiak, Md and Rashid, Nayeeb and Akash, Ahsan Habib and Abrar, Ridwan and Dastider, Ankan Ghosh and Sushmit, Asif Shahriyar and Hasan, Taufiq and others},
  journal={arXiv preprint arXiv:2006.06246},
  year={2020}
}
```
