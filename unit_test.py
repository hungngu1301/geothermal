from matplotlib import pyplot as plt    
from functions import *
from numpy.linalg import norm
import numpy as np

def test_ode_pressure():
	#Test 1 
	dpdt = -32
	dpdt_2 = ode_model_pressure(0,1,2,3,4,5,6,7)
	assert dpdt == dpdt_2
	
	#Test 2
	dpdt = -92
	dpdt_2 = ode_model_pressure(10,9,8,7,6,5,4,3)
	assert dpdt == dpdt_2

def test_ode_concentration():
	#Test 1
	dcdt = 64+(10/3)
	dcdt_2 = ode_model_concentration(0,1,2,3,4,5,6,7,8,9)
	assert dcdt == dcdt_2

	#Test 2
	dcdt = -16-(64/7)
	dcdt_2 = ode_model_concentration(10,9,8,7,6,5,4,3,2,1)
	assert dcdt == dcdt_2

def test_pressure_solver():
	#Test 1: Increasing q
	B = [10,22.5,57.25]
	t,A = solve_ode(ode_model_pressure,0,2,1,10,[[1,2],2,3,4,5,[1,2]])
	for i in range(len(A)): 
		assert A[i]==B[i]
		

	#Test 2: decreasing q
	B = [40,200,997]
	t,A = solve_ode(ode_model_pressure,0,2,1,40,[[2,1],5,4,3,2,[1,2]])
	for i in range(len(A)): 
		assert A[i]==B[i]

	#Constant q
	B = [40, 203, 1018]
	t,A = solve_ode(ode_model_pressure,0,2,1,40,[[3,3],5,4,3,2,[0,0]])
	for i in range(len(A)): 
		assert A[i]==B[i]

	#Test time array
	time=np.array([0, 1, 2])
	t,A = solve_ode(ode_model_pressure,0,2,1,40,[[3,3],5,4,3,2,[0,0]])
	assert norm(t - time) < 1.e-10


def test_concentration_solver():
	#Test 1: constant q
	B = [1, -11, -65]
	t,A = solve_ode_concentration(ode_model_concentration,0,2,1,1,[1,1,1],[1,2,3],[1,1,1,1,1,4,1,3])
	for i in range(len(A)): 
		assert A[i]==B[i]
	

	#Test 2: q changes
	B = [2, -0.5, -20.75]
	t,A = solve_ode_concentration(ode_model_concentration,0,2,1,2,[3,2,3],[1,2,3],[1,1,1,1,1,4,1,3])
	for i in range(len(A)): 
		assert A[i]==B[i]

	#Test 3: decimal numbers
	B = [2, 2.36, 2.5812]
	t,A = solve_ode_concentration(ode_model_concentration,0,2,1,2,[1,1,1],[1,2,3],[1,10,0.1,0.1,1,4,1,3])
	for i in range(len(A)): 
		assert A[i]==B[i]

	#Test 4: decimal numbers 2
	B = [2, 0.125, -4.1875]
	t,A = solve_ode_concentration(ode_model_concentration,0,2,1,2,[1,0,1],[1,1,3],[2,2,0.1,0.1,1,5,1,3])
	for i in range(len(A)): 
		assert A[i]==B[i]

	#Test 5: decimal numbers 2
	time=np.array([0, 1, 2])
	t,A = solve_ode_concentration(ode_model_concentration,0,2,1,2,[1,0,1],[1,1,3],[2,2,0.1,0.1,1,5,1,3])
	assert norm(t-time) < 1.e-10

	#cf_solve_ode_pressure(t, t0, t1, dt, P0, q, dqdt, a, b, c)
	#cf_solve_ode_concentration(t, t0, t1, dt, C0, q_CO2, P, M0, a, b, P0, d):

def test_cf_ode_pressure():
	#Test 1 
	dpdt = -32
	dpdt_2 = cf_ode_model_pressure(1,2,7,5,3,4,6)
	assert dpdt == dpdt_2
	
	#Test 2
	dpdt = -92
	dpdt_2 = cf_ode_model_pressure(9,8,3,5,7,6,4)  
	assert dpdt == dpdt_2
	
def test_cf_ode_concentration():
	#Test 1
	dcdt = 64+(10/3)
	dcdt_2 = cf_ode_model_concentration(0,1,2,6,3,4,5,7,8,9) 
	assert dcdt == dcdt_2

	#Test 2
	dcdt = -16-(64/7)
	dcdt_2 = cf_ode_model_concentration(10,9,8,4,7,6,5,3,2,1) 
	assert dcdt == dcdt_2

def test_cf_pressure_solver():
	#Test 1: Increasing q
	B = [8,6.5,5.25]
	A = cf_solve_ode_pressure(0, 0, 2, 1, 8, [1,2,3], [1,1,1], 1, 1, 1)
	for i in range(len(A)): 
		assert A[i]==B[i]
		

	#Test 2: decreasing q
	B = [8, 8.5, 9]
	A = cf_solve_ode_pressure(0, 0, 2, 1, 8, [3,2,1], [-1,-1,-1], 1, 2, 3)
	for i in range(len(A)): 
		assert A[i]==B[i]

	#Constant q
	B = [8,8,8]
	A = cf_solve_ode_pressure(0, 0, 2, 1, 8, [3, 3, 3], [0, 0, 0], 2, 2, 2)
	for i in range(len(A)): 
		assert A[i]==B[i]

def test_cf_concentration_solver():
	#Test 1: constant q
	B = [2,1.925,1.865]
	A = cf_solve_ode_concentration(0, 0, 2, 1, 2, [3,3,3], [1,2,3], 10, 1, 1, 4, 1)
	for i in range(len(A)): 
		assert A[i]==B[i]
	

	#Test 2: q changes
	B = [4, 4.75, 6.625]
	A = cf_solve_ode_concentration(0, 0, 2, 1, 4, [1,2,4], [1,2,4], 2, 1, 2, 4, 1)
	for i in range(len(A)): 
		assert A[i]==B[i]

	#Test 3: decimal numbers
	B = [4, 5.6875, 13.17578125]
	A = cf_solve_ode_concentration(0, 0, 2, 1, 4, [1.5,3,4.5], [1,2,4], 2, 0.1, 0.2, 4, 1)
	for i in range(len(A)): 
		assert A[i]==B[i]

	#Test 4: decimal numbers 2
	B = [3.5, 3.5, 4.90625]
	A = cf_solve_ode_concentration(0, 0, 2, 1, 3.5, [1.5,3,4.5], [1,2,4], 2, 2, 1, 4, 1)
	for i in range(len(A)): 
		assert A[i]==B[i]
