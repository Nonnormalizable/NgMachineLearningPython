#!/usr/bin/env python

#   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix

# ============= YOUR CODE HERE ==============
# Instructions: Return the 5x5 identity matrix 
#               In octave, we return values by defining which variables
#               represent the return values (at the top of the file)
#               and then set them accordingly. 

from pprint import pprint
import numpy as np

# OK, there's no "the" identity matrix in Python, but a numpy array of arrays
# seems the appropiate thing in this context.
A = np.identity(5)
pprint(A)

# Print, return, whatever.



# ===========================================

