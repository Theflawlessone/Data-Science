# Image Processing Report
In this report, we will document the steps involved in processing an image, including resizing, grayscale conversion, and convolution with random filters. We will use the code provided and visualize the outputs at each step.

You can find the code and the entire process in this Google Colab Notebook[link](https://colab.research.google.com/drive/17CntMWGmxQa0gnsV-BJiu1CEo9Ei_--r?usp=sharing).
## Load RGB Image from URL
The process begins by loading an RGB image from a given URL using the imageio library. The loaded image is displayed using matplotlib.

## Resize Image
The loaded image is resized to a specific size (224x224x3) to prepare it for further processing. The resized image is displayed.
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/cd2b8165-12be-4a3c-92f0-83260f7b5a2f)
## Grayscale Image Conversion
The image is converted to grayscale using the skimage library, specifically the color.rgb2gray function. The first channel of the image is displayed as an example of a grayscale image.

The size of the resulting grayscale image is checked, and it is confirmed to be 224x224.
## Convolve with Random Filters
The image is then convolved with 10 random n x n filters. The code uses the scipy.signal.convolve2d function for this purpose. Both the feature maps and the corresponding filters are displayed side-by-side.
