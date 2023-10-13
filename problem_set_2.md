# Problem Set Two 
In this report, we will document the steps involved in processing an image, including resizing, grayscale conversion, and convolution with random filters. We will use the code provided and visualize the outputs at each step.

You can find the code and the entire process in this Google Colab Notebook [problem_set_2.ipynb](https://colab.research.google.com/drive/15XaUw0xPjNifa0bSiKIuTS3C53Xv_vKB?usp=sharing).

## Load RGB Image from URL
The process begins by loading an RGB image from a given URL using the imageio library. The loaded image is displayed using matplotlib.
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/cf5b1cf5-c68c-48f8-a4c0-db02ae118a2b)
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/480ab75f-90f5-4185-9605-9be123fabe5d)
### Code
``python plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("Loaded Image")
plt.show()``

## Resize Image
The loaded image is resized to a specific size (224x224x3) to prepare it for further processing. The resized image is displayed.
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/bb680857-aea5-4996-bd73-91946b6a786a)
### Code
``python new_size = (224, 224)
resized_image = image.resize(new_size)``

## Grayscale Image Conversion
The image is converted to grayscale using the skimage library, specifically the color.rgb2gray function. The first channel of the image is displayed as an example of a grayscale image. The size of the resulting grayscale image is checked, and it is confirmed to be 224x224.
![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/ebb6a1e3-d6be-41f3-8952-45f1778b7fb3)
### Code
``python plt.imshow(grayscale_image, cmap='gray')
plt.axis("off")
plt.show()``

## Convolve with Random Filters
The image is then convolved with 10 random n x n filters. The code uses the scipy.signal.convolve2d function for this purpose. Both the feature maps and the corresponding filters are displayed side-by-side.

![image](https://github.com/Theflawlessone/Data-Science/assets/142954344/6dfd9352-9b19-4252-9a21-070a704c1d88)
### Code
```python for i in range(9):
    a = 2*np.random.random((3,3))-1
    print(a)
    z=conv2(x,a)
    plot(z)    ```
