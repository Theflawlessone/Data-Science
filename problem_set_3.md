# Report on Convolutional Neural Network Feature Visualization
This report aims to visualize the features learned by a convolutional layer in a CNN, specifically the AlexNet model. We will select an image from the dataset, analyze the feature maps produced by applying convolutional filters to it, and discuss the process of loading a pretrained model, finetuning it, and visualizing the results.
## Data Loading and Visualization
First, we download the Flowers 102 dataset and perform necessary preprocessing, including loading the dataset labels and extracting the images.
``python # Downloading and extracting the dataset
# Uncomment the following lines if you are running this in a Jupyter Notebook
!wget https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt
!wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip 'flower_data.zip'``
We randomly select an image from the dataset (e.g., at index 50) and display it along with its corresponding label.
`python i = 50
plot(images[i],dataset_labels[i]);``
