import numpy as np

# From a Python list:
my_list = [1, 2, 3, 4, 5]
my_array = np.array(my_list)

## Creating NumPy Arrays:
arr = np.array([1, 2, 3])
zeros = np.zeros((2, 3))  # Creates a 2x3 array filled with zeros
Example zeros = np.zeros(5)  # Creates [0. 0. 0. 0. 0.]
ones = np.ones((3, 2))    # Creates a 3x2 array filled with ones
Example ones = np.ones(5)    # Creates [1. 1. 1. 1. 1.]
random_array = np.random.rand(3, 3)  # Creates a 3x3 array with random values between 0 and 1
range_array = np.arange(0, 10, 2)  # Creates an array with values [0, 2, 4, 6, 8]
linspace_array = np.linspace(0, 1, 5)  # Creates an array with 5 evenly spaced values between 0 and 1

### Array Operations:
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b

#### Element-wise square root
result = np.sqrt(my_array)

##### Broadcasting
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
result = array1 + array2  # Adds corresponding elements together

###### Array Indexing and Slicing:
arr = np.array([0, 1, 2, 3, 4, 5])
print(arr[2])        # Accessing a single element: 2
print(arr[1:4])      # Slicing: [1, 2, 3]
my_array[0]        # Access the first element (zero-based index)
my_array[1:4]      # Slice elements from index 1 to 3 (exclusive)
my_array[2:]       # Slice elements from index 2 to the end


####### Array Shape and Reshaping:
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)           # Shape of the array: (2, 3)
reshaped = arr.reshape(3, 2)

######## Array Functions:
NumPy provides many mathematical functions that operate on arrays, such as np.sum, np.mean, np.max, np.min
arr = np.array([1, 2, 3, 4, 5])
sum_of_elements = np.sum(arr)
mean_value = np.mean(arr)
max_value = np.max(arr)
np.min(my_array)        # Minimum element
np.std(my_array)        # Standard deviation
np.sort(my_array)       # Sorts the elements in ascending order
np.unique(my_array)     # Returns unique elements

######### Array Attributes:
my_array.shape     # Returns the dimensions of the array as a tuple
my_array.dtype     # Returns the data type of the elements in the array
my_array.ndim      # Returns the number of dimensions (axes) in the array
my_array.size      # Returns the total number of elements in the array

########## Creating Identity Matrices:
The np.eye() function is used to create an identity matrix, which is a square matrix with ones on the diagonal and zeros elsewhere. The identity matrix is often denoted as "I" in mathematics.
print(identity_matrix)
identity_matrix = np.eye(3) # Create a 3x3 identity matrix

########### Transposing Arrays:
The np.transpose() function is used to transpose a NumPy array. Transposing an array means flipping its rows and columns. It's particularly useful when you need to change the dimensions or shape of your data.
print(transposed_array)
transposed_array = np.transpose(original_array) # Transpose the array
Both np.transpose() and .T perform the same operation of transposing the array. Transposing is especially useful when dealing with matrix operations or when you need to change the orientation of your data for specific calculations or applications.
transposed_array = original_array.T




