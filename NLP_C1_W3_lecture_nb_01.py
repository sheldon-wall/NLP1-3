import numpy as np

# Defining lists and numpy arrays
alist = [1, 2, 3, 4, 5]   # Define a python list. It looks like an np array
narray = np.array([1, 2, 3, 4]) # Define a numpy array

print(alist)
print(narray)

print(type(alist))
print(type(narray))

# Algebraic operators on Numpy arras vs. Python lists
print(narray + narray)
print(alist + alist)

print(narray * 3) # scaling the vector
print(alist * 3) # replicating the list 3 times

okmatrix = np.array([[1, 2], [3,4]])
print(okmatrix)
print(okmatrix * 2)

# Scale by 2 and translate 1 unit
result = okmatrix * 2 + 1
print(result)

# subtract two sum compatible matrices. THis is called the difference vector
result = (okmatrix * 2) - okmatrix
print(result)

result = okmatrix * okmatrix
print(result)

# Transpose a matrix
matrix3x2 = np.array([[1, 2], [3, 4], [5, 6]])
print('Original matrix 3 x 2\n', matrix3x2)
print('Transposed matrix 2 x 3\n', matrix3x2.T)

nparray = np.array([[1, 2, 3, 4]]) # Define a 1 x 4 matrix. Note the 2 level of square brackets
print('Original array')
print(nparray)
print('Transposed array')
print(nparray.T)

# Get the norm of a nparray or matrix
nparray1 = np.array([1, 2, 3, 4]) # Define an array
norm1 = np.linalg.norm(nparray1)

nparray2 = np.array([[1, 2], [3, 4]]) # Define a 2 x 2 matrix. Note the 2 level of square brackets
norm2 = np.linalg.norm(nparray2)

print(norm1)
print(norm2)

#
nparray1 = np.array([0, 1, 2, 3]) # Define an array
nparray2 = np.array([4, 5, 6, 7]) # Define an array
#
print(np.dot(nparray1, nparray2))
print(nparray1 @ nparray2)

nparray1 = np.array([1, 2, 3, 4]) # Define an array
print(np.linalg.norm(nparray1))
# norm is square root of the dot product with itself
print(np.sqrt(nparray1 @ nparray1))

# Get the mean by rows and columns
nparray2 = np.array([[1, -1], [2, -2], [3, -3]]) # Define a 3 x 2 matrix. Chosen to be a matrix with 0 mean
print(np.mean(nparray2)) # Get the mean for the whole matrix
print(np.mean(nparray2, axis=0)) # Get the mean for each column. Returns 2 elements
print(np.mean(nparray2, axis=1)) # get the mean for each row. Returns 3 elements

# Centering a matrix
nparray2 = np.array([[1, 1], [2, 2], [3, 3]]) # Define a 3 x 2 matrix.
nparrayCentered = nparray2 - np.mean(nparray2, axis=0) # Remove the mean for each column
print(nparray2)
print(nparrayCentered)
print('New mean by column')
print(nparrayCentered.mean(axis=0))

# the euclidean distance for n-dimensional vectors (v and w)
# is the norm of (vector v - vector w)

v = np.array([1, 6, 8])
w = np.array([0, 4, 6])
dist = np.linalg.norm(v - w)
print(f'the Euclidean distance between v and w is {dist:.2f}')

Food = np.array([5, 15])
Agric = np.array([20, 40])
Hist = np.array([30, 20])

cos_sim = np.dot(Agric, Hist) / (np.linalg.norm(Agric) * np.linalg.norm(Hist))
print('cosine similarity =', cos_sim)

USA = np.array([5,6])
Wash = np.array([10,5])
diff = Wash - USA
print(diff)

Russia = np.array([5, 5])
Moscow = np.array([9.3])

Japan = np.array([4,3])
Tokyo = np.array([8.5,2])

Turkey = np.array([3,1])
Ankara = np.array([8.5,.9])

print(Ankara - diff)
print(np.linalg.norm(Turkey + diff - Ankara))
print(np.linalg.norm(Japan + diff - Ankara))

