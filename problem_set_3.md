# Report on Convolutional Neural Network Feature Visualization
This report aims to visualize the features learned by a convolutional layer in a CNN, specifically the AlexNet model. We will select an image from the dataset, analyze the feature maps produced by applying convolutional filters to it, and discuss the process of loading a pretrained model, finetuning it, and visualizing the results.
## Data Loading and Visualization
First, we download the Flowers 102 dataset and perform necessary preprocessing, including loading the dataset labels and extracting the images.
```python
import matplotlib.pyplot as plt
# Define a function for plotting images
def plot(x, title=None):
    x_np = x.cpu().numpy()
    if x_np.shape[0] == 3 or x_np.shape[0] == 1:
        x_np = x_np.transpose(1, 2, 0)
    if x_np.shape[2] == 1:
        x_np = x_np.squeeze(2)
    x_np = x_np.clip(0, 1)
    fig, ax = plt.subplots()
    if len(x_np.shape) == 2:
        im = ax.imshow(x_np, cmap='gray')
    else:
        im = ax.imshow(x_np)
    plt.title(title)
    ax.axis('off')
    fig.set_size_inches(10, 10)
    plt.show()

# Download and extract the dataset
# Uncomment the following lines if you are running this in a Jupyter Notebook
!wget https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt
!wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip 'flower_data.zip'

import torch
from torchvision import datasets, transforms
import os
import pandas as pd

# Directory and transforms
data_dir = '/content/flower_data/'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transform)
dataset_labels = pd.read_csv('Oxford-102_Flower_dataset_labels.txt', header=None)[0].str.replace("'", "").str.strip()

# Load the dataset into a DataLoader for batching
dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False) 

# Extract the batch of images and labels
images, labels = next(iter(dataloader))

# Select an image and plot it with class label
i = 50
plot(images[i], dataset_labels[i]

We randomly select random images from the dataset and display it along with its corresponding label.
python i = 50
plot(images[i],dataset_labels[i]);
```
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/7d6949de-b242-414d-b5d8-421852b47f03)



## Pretrained Model Loading
We load the pre-trained AlexNet model, which includes pre-trained weights. We then run the model on a batch of data from the Flowers 102 dataset. The model's predicted classes are shown along with the images.
```python
import torch
from torchvision import models, transforms
import requests
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define alexnet model
alexnet = models.alexnet(pretrained=True).to(device)
labels = {int(key): value for (key, value) in requests.get('https://s3.amazonaws.com/mlpipes/pytorch-quick-start/labels.json').json().items()}

# Transform image for use in the model
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Select an image
img = images[i]

from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
img = to_pil(img) 

img_t = preprocess(img).unsqueeze_(0).to(device)

# Classify the image with alexnet
scores, class_idx = alexnet(img_t).max(1)
print('Predicted class:', labels[class_idx.item()])
```

## Finetuning
In the process of finetuning, we replace the last layer of the AlexNet with a new final layer that has the appropriate number of outputs to match the Flowers 102 dataset. The network is then trained on the Flowers 102 dataset to achieve accuracy.
```python
# Modify the final fully connected layer
w0 = alexnet.features[0].weight.data
w1 = alexnet.features[3].weight.data
w2 = alexnet.features[6].weight.data
w3 = alexnet.features[8].weight.data
w4 = alexnet.features[10].weight.data
w5 = alexnet.classifier[1].weight.data
w6 = alexnet.classifier[4].weight.data
w7 = alexnet.classifier[6].weight.data
# Save and Load
# w = [w0,w1,w2,w3,w4,w5,w6,w7]
# torch.save(w, 'Hahn_Alex.pt')
# w = torch.load('Hahn_Alex.pt')
# [w0,w1,w2,w3,w4,w5,w6,w7] = w
# [w0,w1,w2,w3,w4,w5,w6,w7] = torch.load('Hahn_Alex.pt')
```
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/43d9d188-67aa-47b8-8c24-cb22bb71d243)

## Results Visualization
We calculate the accuracy on both the training and validation data. Additionally, we show a sample of images with their true labels and the predicted labels.
```python
# Extract feature maps using F.conv2d
f0 = F.conv2d(img_t, w0, stride=4, padding=2)
# Visualize filters
for i in range(64):
    tensor_plot(w0, i)
    plt.imshow(f0[0, i, :, :].cpu().numpy())
    plt.show()
# Visualize feature maps with filters
plot_feature_maps_with_filters(f0, w0)
```
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/de744ae7-8144-40f3-bf75-32cd4742534f)
