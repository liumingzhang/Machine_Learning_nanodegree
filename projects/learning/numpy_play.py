# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

hw12 = '%s %s %d' % (hello, world, 12)
#print (hw12)

s = '  work  '
s.strip()

list = [1,2,3,5,'foo']
list.append('derp')
list.pop()

nums = [0,1,2,3,4,5]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
even_squares_dic = {x : x ** 2 for x in nums if x % 2 == 0}


d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del d['fish']         # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"

d = {(x,x+1): x for x in range(10)} # Create a dictionary with tuple keys

import numpy as np

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"

a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3] #b is not a copy of a, they share the same data

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
"""
Integer array indexing: When you index into numpy arrays using slicing, the 
resulting array view will always be a subarray of the original array. In contrast, 
integer array indexing allows you to construct arbitrary arrays using the data from another array. 
Here is an example:
"""
                   
a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"

"""
One useful trick with integer array indexing is selecting or mutating one element 
from each row of a matrix:
"""
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
b = np.array([0,2,0,1])
print (a[np.arange(4), b])
a[np.arange(4), b] += 10
print(a)

"""
print elements that greater than 2
"""
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print (a[a > 2])

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

print (x * y)
print (np.sqrt(x))

"""
矩阵乘法用dot
"""
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
print(v.shape)
w = np.array([11, 12])

print(np.dot(v,w))
print(np.dot(x,v))

x = np.array([[1,2],[3,4]])
print(np.sum(x))
print(np.sum(x, axis=0))# Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))# Compute sum of each row; prints "[3 7]"

"""
Transpose,转置,a(i,j) = b(j,i)
"""
x = np.array([[1,2,3],[4,5,6]])
print(x)
y = np.transpose(x)
print(y)

"""
reshape and broadcasting
"""
v = np.array([1,2,3])
v1 = np.reshape(v,(3,1))
print(v1)
w = np.array([4,5])
print(v1 * w)

from scipy.spatial.distance import pdist, squareform








