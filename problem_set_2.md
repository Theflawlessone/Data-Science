# Image Processing Report
In this report, we will document the steps involved in processing an image, including resizing, grayscale conversion, and convolution with random filters. We will use the code provided and visualize the outputs at each step.

You can find the code and the entire process in this Google Colab Notebook [problem_set_2.ipynb](https://colab.research.google.com/drive/17CntMWGmxQa0gnsV-BJiu1CEo9Ei_--r?usp=sharing).

## Load RGB Image from URL
The process begins by loading an RGB image from a given URL using the imageio library. The loaded image is displayed using matplotlib.
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/cf5b1cf5-c68c-48f8-a4c0-db02ae118a2b)
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/480ab75f-90f5-4185-9605-9be123fabe5d)
### Code
'import numpy as np
from skimage import io

image_url = "https://media.istockphoto.com/id/173240099/photo/surprise-kitty-cute-black-cat-screaming.jpg?s=612x612&w=0&k=20&c=fKCBMfIQuunPUC0DQTcI25iFnBAEaCfLxZX94oajjNM="
image = io.imread(image_url)'

## Resize Image
The loaded image is resized to a specific size (224x224x3) to prepare it for further processing. The resized image is displayed.
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/bb680857-aea5-4996-bd73-91946b6a786a)
### Code
'plt.imshow(image)
plt.title("Resized Image (224x224)")
plt.axis("off")
plt.show()'

## Grayscale Image Conversion
The image is converted to grayscale using the skimage library, specifically the color.rgb2gray function. The first channel of the image is displayed as an example of a grayscale image. The size of the resulting grayscale image is checked, and it is confirmed to be 224x224.
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/ebb6a1e3-d6be-41f3-8952-45f1778b7fb3)
### Code
'plt.imshow(first_channel, cmap="gray")
plt.title("First Channel (Red, for example)")
plt.axis("off")
plt.show()'

## Convolve with Random Filters
The image is then convolved with 10 random n x n filters. The code uses the scipy.signal.convolve2d function for this purpose. Both the feature maps and the corresponding filters are displayed side-by-side.

![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/6dfd9352-9b19-4252-9a21-070a704c1d88)
### Code
'for i in range(9):
    a = 2*np.random.random((3,3))-1
    print(a)
    z=conv2(x,a)
    plot(z) '
