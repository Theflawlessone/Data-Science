# Report on Convolutional Neural Network Feature Visualization
This report aims to visualize the features learned by a convolutional layer in a CNN, specifically the AlexNet model. We will select an image from the dataset, analyze the feature maps produced by applying convolutional filters to it, and discuss the process of loading a pretrained model, finetuning it, and visualizing the results.
## Data Loading and Visualization
First, we download the Flowers 102 dataset and perform necessary preprocessing, including loading the dataset labels and extracting the images.
`python # Downloading and extracting the dataset
# Uncomment the following lines if you are running this in a Jupyter Notebook
!wget https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt
!wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip 'flower_data.zip'`

We randomly select random images from the dataset and display it along with its corresponding label.
`python i = 50
plot(images[i],dataset_labels[i]);`
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/7d6949de-b242-414d-b5d8-421852b47f03)

## Pretrained Model Loading
We load the pre-trained AlexNet model, which includes pre-trained weights. We then run the model on a batch of data from the Flowers 102 dataset. The model's predicted classes are shown along with the images.
``python 

## Finetuning
In the process of finetuning, we replace the last layer of the AlexNet with a new final layer that has the appropriate number of outputs to match the Flowers 102 dataset. The network is then trained on the Flowers 102 dataset to achieve accuracy.
``python

## Results Visualization
We calculate the accuracy on both the training and validation data. Additionally, we show a sample of images with their true labels and the predicted labels.
``python
