# ENGSCI233: Lab - Sampled Data
# sdlab_functions.py

# PURPOSE:
# To IMPLEMENT cubic spline interpolation.

# PREPARATION:
# Notebook sampling.ipynb, ESPECIALLY Section 1.3.1 theory of cubic splines.

# SUBMISSION:
# - YOU MUST submit this file to complete the lab. 
# - DO NOT change the file name.

# TO DO:
# - COMPLETE the functions spline_coefficient_matrix(), spline_rhs() and spline_interpolation().
# - COMPLETE the docstrings for each of these functions.
# - TEST each method is working correctly by passing the asserts in sdlab_practice.py.
# - DO NOT modify the other functions.

import numpy as np


def spline_coefficient_matrix(xi):	
	'''
		Creates a Spline Coefficent Matrix for a set of points.

		Parameters
		----------

		xi : np.array
			Vector containing all x-values for the points. Must be of dimension 1xn.

		Returns
		-------

		A : np.array
			Matrix containing coefficients for all sub intervals for cubic splines. 

	'''
	# Creating a matrix full of 0s of dimension (4N-4, 4N-4).
	A = np.zeros((4*(len(xi)-1),4*(len(xi)-1)))
		
	# Looping through subintervals and set appropriate x0 for each of them. 
	for i in range(len(xi)-1):
		x_0 = xi[i]

		# At each subinterval, compute boundary conditions and 
		# store co-efficients in A at appropriate locations. 
		for j in range(i,i+2):
			a_vals = np.zeros(4)
			a_vals[0] = 1
			a_vals[1] = (-x_0 + xi[j])
			a_vals[2] = (-x_0 + xi[j])**2
			a_vals[3] = (-x_0 + xi[j])**3
			A[j+i][4*i:(4*i+4)]= a_vals
	
	# Looping through subintervals and set appropriate x0 for each of them. 
	for i in range(1,len(xi)-1):
		x_0 = xi[i-1]

		# Compute co-efficients for first derivative conditions and store them appropriately in A, 
		# after boundary conditions.
		da_vals = np.zeros(8)
		da_vals[1]= 1
		da_vals[2]= 2*(-x_0 + xi[i])
		da_vals[3]= 3*(-x_0 + xi[i])**2
		da_vals[5]= -1
		A[i+(2*len(xi))-3][4*(i-1):4*(i-1)+8]= da_vals

	# Looping through subintervals and set appropriate x0 for each of them, excluding first and last point.
	for i in range(1,(len(xi)-1)):
		x_0 = xi[i-1]

		# Compute co-efficients for second derivative conditions and store them appropriately in A,
		# after the first derivative conditions.
		d2a_vals = np.zeros(8)
		d2a_vals[2] = 2
		d2a_vals[3] = 6*(-x_0 + xi[i])
		d2a_vals[6] = -2
		A[i+(3*len(xi))-5][4*(i-1):4*(i-1)+8] = d2a_vals

	# Compute co-efficients for second derivative condition at first point, in first subinterval.
	end1a = np.zeros(4)
	end1a[2] = 2

	# Store first point coefficients in the second to last row of A, at appropriate column locations.
	A[-2][0:4] = end1a
	
	# Compute co-efficients for second derivative condition at last point, in last subinterval.
	end2a = np.zeros(4)
	end2a[2] = 2
	end2a[3] = 6*(-xi[-2]+xi[-1])

	# Store last point coefficients in the last row of A, at appropriate column locations.
	A[-1][-4::] = end2a
	
	return A


def spline_rhs(xi, yi):
	''' Creates the rhs vector for the spline co-efficient matrix

	Parameters
	----------

	xi: np.array
		A 1xn vector containing all x values for the points.

	yi: np.array
		A 1xn vector containing all y values for the points. 

	Returns
	-------

	b: np.array
		A 1xn vector containing all rhs values for the cubic spline matrix. 

	'''
	# Create an array of zeros with length 4N - 1.
	b = np.zeros(4*(len(yi)-1))

	# First element of rhs will always be the same as the first y.
	b[0] = yi[0]

	# (2n - 2)nd element of rhs will always be the same as the last y. 
	b[2*(len(yi)-1)-1]= yi[-1]

	# All elements between first and (2n - 2)nd element will be y elements between 
	# first and last element, repeated twice. 
	mid_i = (2*(len(yi)-1)-2)
	count = 1
	for i in range(1,mid_i,2):
		b[i] = yi[count]
		b[i+1] = yi[count]
		count += 1
 
	return b
	

def spline_interpolate(xj, xi, ak):
	''' Interpolates a set of values according to the cubic spline coefficients

		Parameters
		----------

		xj: np.array
			A 1xn vector containing x-values of points over which interpolation needs to occur. 
		
		xi: np.array
			A 1xn vector containing x-values of points using which the cubic spline coeffiecients
			were calculated.

		ak: np.array
			A 1xn vector containing the cubic spline co-efficients. 

		
		Returns
		-------

		yj: np.array
			A 1xn vector containing all the y-values for the interpolated points. 

		Notes
		-----
		- ak is not the same as the cubic spline coefficient matrix. It can be obtained using
		the coeffecient matrix and the rhs vector and solving them. 

		- Due to the nature of the algorithm, it is NOT essential for values in xj to be in ascending order. 
	'''

	# Create an array of zeros for the new x interval. 
	yj = np.zeros(len(xj))

	# Looping through each point in the new x interval.
	for i in range(len(xj)):
		j = 0

		# Checking if point is in the range of existing data. Else move onto next value. 
		if xj[i] >= xi[0]:
			if xj[i]<= xi[-1]:

				# Finding the appropriate subinterval for that x value.
				while xj[i] >= xi[j]:
					if j < len(xi)-1:
						j += 1
					else:
						break
				
				# Calculating interpolated values by accessing appropriate coefficents for the subinterval.
				yj[i] = ak[4*(j-1)] + (ak[(4*(j-1))+1]*(xj[i]-xi[j-1])) + (ak[(4*(j-1))+2]*((xj[i]-xi[j-1])**2)) + (ak[(4*(j-1)+3)]*((xj[i]-xi[j-1])**3))
			else:
				pass
		else:
			pass
	
	return yj
	
	
	
	
	
		
	
# this function is complete
def display_matrix_equation(A,b):
	''' Prints the matrix equation Ax=b to the screen.
	
		Parameters
		----------
		A : np.array
			Matrix.
		b : np.array
			RHS vector.
			
		Notes
		-----
		This will look horrendous for anything more than two subintervals.	
	'''
	
	# problem dimension
	n = A.shape[0]
	
	# warning
	if n > 8:
		print('this will not format well...')
		
	print(' _'+' '*(9*n-1) +'_  _       _   _        _')
	gap = ' '
	for i in range(n):
		if i == n - 1:
			gap = '_'
		str = '|{}'.format(gap)
		str += ('{:+2.1e} '*n)[:-1].format(*A[i,:])
		str += '{}||{}a_{:d}^({:d})'.format(gap,gap,i%4,i//4+1)+'{}|'.format(gap)
		if i == n//2 and i%2 == 0:
			str += '='
		else:
			str += ' '
		str += '|{}{:+2.1e}{}|'.format(gap,b[i],gap)
		print(str)
	
# this function is complete
def get_data():
	# returns a data vector used during this lab
	xi = np.array([2.5, 3.5, 4.5, 5.6, 8.6, 9.9, 13.0, 13.5])
	yi = np.array([24.7, 21.5, 21.6, 22.2, 28.2, 26.3, 41.7, 54.8])
	return xi,yi
		
# this function is complete
def ak_check():
	# returns a vector of predetermined values
	out = np.array([2.47e+01, -4.075886048665986e+00,0.,8.758860486659859e-01,2.15e+01,
		-1.448227902668027e+00,2.627658145997958e+00,-1.079430243329928e+00,2.16e+01,
		5.687976593381042e-01,-6.106325839918264e-01,5.358287012458253e-01,2.22e+01,
		1.170464160078432e+00,1.157602130119396e+00,-2.936967278262911e-01,2.82e+01,
		1.862652894849505e-01,-1.485668420317224e+00,1.677900564431842e-01,2.63e+01,
		-2.825777017172887e+00,-8.312872001888050e-01,1.079137281294699e+00,4.17e+01,
		2.313177016138269e+01,9.204689515851896e+00,-6.136459677234598e+00])
	return out
	
# this function is complete
def polyval(a,xi):
	''' Evaluates a polynomial.
		
		Parameters
		----------
		a : np.array
			Vector of polynomial coefficients.
		xi : np.array
			Points at which to evaluate polynomial.
		
		Returns
		-------
		yi : np.array
			Evaluated polynomial.
			
		Notes
		-----
		Polynomial coefficients assumed to be increasing order, i.e.,
		
		yi = Sum_(i=0)^len(a) a[i]*xi**i
		
	'''
	# initialise output at correct length
	yi = 0.*xi
	
	# loop over polynomial coefficients
	for i,ai in enumerate(a):
		yi = yi + ai*xi**i
		
	return yi