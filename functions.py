import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import interpolate


def ode_model_pressure(t, P, q, a, b, P0, c, dqdt):
	''' Return the derivative dP/dt at time, t, for given parameters.

		Parameters:
		-----------
		t : float
			Independent variable.
		P : float
			Dependent variable.
		q : float
			Source/sink rate.
		a : float
			Source/sink strength parameter.
		b : float
			Recharge strength parameter.
		P0 : float
			Ambient value of dependent variable.
		c : float
			slow drainage parameter
		dqdt: float
			mass rate differential
		Returns:
		--------
		dxdt : float
			Derivative of dependent variable with respect to independent variable.

		Notes:
		------
		None

		Examples:
		---------
		>>> ode_model_pressure(0, 1, 2, 3, 4, 5, 6, 7)
		-32

	'''
	
	dPdt=-a*q-b*(P-P0)-c*dqdt
	return dPdt

def ode_model_concentration(t,C,q,M,a,b,P,P0,d,C0):
	'''returns dCdt at given time and parameters
	
	Parameters:
	t: float
	   Independent variable
	C: concentration 
	   Dependent variable
	q: float
	   Injection rate of CO2
	M: float
	   Ambient mass of the system 
	a: float
	   Source/sink strength parameter
	b: float 
	   Recharge strength parameter.
	P: float
	   Pressure of the system at time t
	P0: float
	   Ambient pressure of the system 
	d: float
	   CO2 reaction rate 
	C0: float
		Ambient concentration of the system
	
	Outputs:
	dCdt: float
		  rate of change of concentration of CO2 at time t 

	Notes: 
	N/A
	'''

	if P>P0:
		return (1-C)*q/(M)-d*(C-C0)
	else:
		return (1-C)*q/(M)-b*(P-P0)*(C0-C)/(M*a)-d*(C-C0)

def solve_ode_concentration(f, t0, t1, dt, x0, q, P, pars):
	'''
	solve the concentration ODE by using improved euler on ode_model_concentration()

	Parameters:
	f : callable
			Function that returns dxdt given variable and parameter inputs.
	t0 : float
		Initial time of solution.
	t1 : float
		Final time of solution.
	dt : float
		Time step length.
	x0 : floar
		Initial value
	q : array-like 
		interpolated qCO2
	P : array-like
		solved P
	pars: array-like
		parameter inputs

	Outputs:
	t: array-like
	   time
	x: array-like
	   concentration

	Notes: 
	Assume the parameters in pars are in this order
		1. q: injection rate [kg/s]
		2. M: mass of the system []
		3. a: source/sink coefficient 
		4. b: recharge coefficient 
		5. P: pressure of the system at time t []
		6. P0: Ambient pressure []
		7: d: CO2 reaction rate coefficient []
		8: C0: initial concentration of CO2 
	'''

	#Allocate time and concentration array 
	nx = int(np.ceil((t1-t0)/dt))
	t = t0+np.arange(nx+1)*dt
	x=0.*t
	x[0]=x0

	#Set up initial injection rate and pressure 
	pars[0]=q[0]
	pars[4]=P[0]
	
	#Improved Euler 
	for i in range(1,nx+1):
		x_predict=x[i-1]+dt*f(t[i-1],x[i-1],*pars)
		x[i]=x[i-1]+(f(t[i-1],x[i-1],*pars)+f(t[i-1]+dt,x_predict,*pars))*dt/2

		#Update the parameters 
		pars[0]=q[i]
		pars[4]=P[i]

	return t,x

def solve_ode(f, t0, t1, dt, P, pars):
	''' Solve the pressure ODE numerically.

		Parameters:
		-----------
		f : callable
			Function that returns dxdt given variable and parameter inputs.
		t0 : float
			Initial time of solution.
		t1 : float
			Final time of solution.
		dt : float
			Time step length.
		P : float
			Initial value of solution.
		pars : array-like
			List of parameters passed to ODE function f.

		Returns:
		--------
		t : array-like
			Independent variable solution vector.
		x : array-like
			Dependent variable solution vector.

		Notes:
		------
		ODE should be solved using the Improved Euler Method. 

		Function q(t) should be hard coded within this method. Create duplicates of 
		solve_ode for models with different q(t).

		Assume that ODE function f takes the following inputs, in order:
			1. independent variable
			2. dependent variable
			3. forcing term, q
			4. all other parameters
		
		The parameters are in order:
			1. q: forcing term (array-like)
			2. a: source/sink parameter
			3. b: recharge parameter
			4. P0: initial pressure
			5. c: slow drainage
			6. dqdt: rate of change of forcing terms

	'''

	#Pre allocate time 
	nx = int(np.ceil((t1-t0)/dt))
	t = t0+np.arange(nx+1)*dt

	#Preallocate value for first iteration
	x=0.*t
	x[0]=P
	q=pars[0]
	dqdt=pars[5]

	#Loop 
	for i in range(1,nx+1):
		#Update parameters
		pars[0]=q[i-1]
		pars[5]=dqdt[i-1]

		#Prediction step 
		x_predict=x[i-1]+dt*f(t[i-1],x[i-1],*pars)

		#Correction
		x[i]=x[i-1]+(f(t[i-1],x[i-1],*pars)+f(t[i-1]+dt,x_predict,*pars))*dt/2
	return t,x

def pressure_analytical(t,P0,a,b,c):
	'''returns the analytical solution of the pressure model
	
	Parameters: 
	t: array-like
		time
	P0: float
		ambient pressure 
	a: float
		source/sink coefficient
	b: float
		recharge coefficient
	c: float
		slow drainage coefficient 
	
	Outputs: 
	P: array-like
		pressure evaluated at time t
	
	Notes: 
	_P and t have the same size 
	Assume that q=50 is constant --> dqdt=0
	

	'''

	#Modify time array so it starts at 0
	t=t-t[0]
	
	#Allocate P array
	P=0.*t
	P[0]=P0
	#Evaluating Pressure using analytical solution 
	for i in range(1,len(t)):
		P[i]=-a*150/b*(1-np.exp(-b*t[i]))+P0
	return P

def concentration_analytical(t,q,M,a,b,P,P0,d,C0):
	'''
	returns the analytical solution for concentration model 

	Parameters: 
	t: array-like
		time 
	q: array-like
		injection rate 
	M: float
		initial mass of the system 
	a: float
		parameter a got from pressure model
	b: float
		papramter b got from pressure model
	P: array-like
		pressure at time t 
	P0: float
		ambient pressure of the system 
	d: float
		parameter d 
	C0: float
		ambient CO2 concentration pf the system

	Outputs: 
	C: array-like
		analytical solution of the concentration model  
	
	'''
	#Modify time array 
	t=t-t[0]
	
	#pre-allocate time array 
	C=0.*t
	for i in range(len(t)):
		if P[i]>P0:
			S=q[i]/M+d
			R=q[i]/M+d*C0
		else: 
			S=q[i]/M-b*(P[i]-P0)/(a*(M))+d
			R=q[i]/M-C0*(b*(P[i]-P0)/(a*(M))-d)
		C[i]=R/S+(C0-R/S)*np.exp(-S*t[i])
	
	return C



def construct_samples_pressure(mean,covariance,N_samples):
	''' This function constructs samples from a multivariate normal distribution
		fitted to the data.

		Parameters:
		-----------
		a : array-like
			Vector of 'a' parameter values.
		b : array-like
			Vector of 'b' parameter values.
		c : array-like
			Vector of 'c' parameter values.
		P : array-like
			Posterior probability distribution.
		N_samples : int
			Number of samples to take.

		Returns:
		--------
		samples : array-like
			parameter samples from the multivariate normal

		Notes:
		-----------
		263 Lab
	'''

	# compute properties (fitting) of multivariate normal distribution
	# mean = a vector of parameter means
	# covariance = a matrix of parameter variances and correlations
	#A, B, C = np.meshgrid(a,b,c,indexing='ij')
	#mean, covariance = fit_mvn([A,B,C], P)

	# 1. create samples using numpy function multivariate_normal (Google it)
	samples=np.random.multivariate_normal(mean, covariance, size=N_samples)

	return samples

def model_ensemble_pressure(samples,q_pressure,dqdt,P_t):
	''' Runs the model for given parameter samples and returns the results.

		Parameters:
		-----------
		samples : array-like
			parameter samples from the multivariate normal
		q_pressure: array-like
			net mass rate 
		dqdt: array-like
			rate of change of net mass rate
		P_t : float
			initial pressure
		
		Returns:
		-----------
		pressure: array-like
			list of results for given parameters samples
	'''

	#Create an empty list to store results
	pressure=[]
	# 3. for each sample, solve and plot the model  (see TASK 1)
	for a,b,c in samples:
		t,pm=solve_ode(ode_model_pressure,1968,2050,0.1,P_t,[q_pressure, a, b, 6.17, c, dqdt])
		pressure.append(pm)

	return pressure


def model_ensemble_concentration(samples,q_injection,pressure):
	''' Runs the concentration model for given parameter samples and gives the results.

		Parameters:
		-----------
		samples : array-like
			parameter samples from the multivariate normal
		q_injection: array-like
			injection rate
		pressure: array-like
			pressure of the system 

		Returns: 
		-----------
		concentration: array-like
			list of results for given parameter samples
	'''

	#Create a list to store results
	concentration=[]

	#for each sample, solve and store the result
	for M,d in samples:
		t,pm=t,pm=solve_ode_concentration(ode_model_concentration, 1980, 2050, 0.1, 0.03, q_injection, pressure, [q_injection[0],M,0.0019741,0.14734525,pressure[0],6.17,d,0.03])
		concentration.append(pm)
	
	return concentration

def construct_samples_concentration(mean,covariance,N_samples):
	''' This function constructs samples from a multivariate normal distribution
		fitted to the data.

		Parameters:
		-----------
		M : array-like
			Vector of 'M' parameter values.
		d : array-like
			Vector of 'd' parameter values.
		P : array-like
			Posterior probability distribution.
		N_samples : int
			Number of samples to take.

		Returns:
		--------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	# compute properties (fitting) of multivariate normal distribution
	# mean = a vector of parameter means
	# covariance = a matrix of parameter variances and correlations

	#create samples using numpy function multivariate_normal (Google it)
	samples=np.random.multivariate_normal(mean, covariance, size=N_samples)
	
	return samples


# CALIBRATION ODE/SOLVE FUNCTIONS
#
# note:
# -----
# due to specific nature of the curve_fit function, different implementations of ode functions to take
# the required inputs and output the desired result were required so not as to destroy the rest of our code

import numpy as np


def cf_ode_model_pressure(P, q, dqdt, P0, a, b, c):
    ''' Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        P : float
            Dependent variable.
        q : float
            Source/sink rate.
        dqdt: float
             Injection/Extraction rate differential.
        P0 : float
            Ambient value of dependent variable.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        c : float
            Slow drainage parameter.
        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        q = q_water - q_CO2

        Examples:
        ---------
        >>> ode_model_pressure(0, 1, 2, 3, 4, 5, 6, 7)
        -20

    '''

    return -a*q-b*(P-P0)-c*dqdt


def cf_ode_model_concentration(t, C, q_CO2, P, Mo, a, b, P0, d, C0):
    ''' Return dC/dt at given time, t, for given parameters.
    
    Parameters:
    -----------
    t: float
       Independent variable.
    C: concentration 
       Dependent variable.
    qCO2: float
       Injection rate of CO2.
    P: float
       Pressure of the system at time t.
    Mo: float
       Initial mass of the system (does not deviate that much)
    a: float
       Source/sink strength parameter.
    b: float 
       Recharge strength parameter.
    P0: float
       Initial/outside pressure of the system. 
    d: float
       CO2 reaction rate.
    C0: float
        Initial concentration of the system.
    
    Outputs:
    --------
    dCdt: float
        Rate of change of concentration of CO2 at time t.

    Notes:
    ------
    C0 is approximatley 3 wt% CO2 for Ohaaki Geothermal System (i.e C0 = 0.03).
    If P>Po the second term goes to zero, hence the if statement.

    '''

    # check condition and use appropriate version of model accordingly 
    if P>P0:
        return (1-C)*(q_CO2/Mo)-d*(C-C0)
    else:
        return (1-C)*(q_CO2/Mo)-(b/(a*Mo))*(P-P0)*(C0-C)-d*(C-C0)


def cf_solve_ode_concentration(t, t0, t1, dt, C0, q_CO2, P, M0, a, b, P0, d):
    ''' Solve the concentration ODE by using improved euler on ode_model_concentration()

    Parameters:
    -----------
        t : array-like
            Independent variable (time in years)
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        C0 : float
            Initial value of solution.
        q_CO2 : array-like
            Injection rates.
        P : array-like
            Pressure values (from model).
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        P0 : float
            Initial pressure value.
        d : float
            CO2 reaction rate.

    Outputs:
    --------
    C: array-like
       Concentration solution array.

    Notes: 
    ------
        The t parameter input is not used- it's purpose is soley to ensure compatibility with
        curve_fit by absorbing the indeprendent variable input from curve_fit.

    '''

    # initialise
    steps = int(np.ceil((t1-t0)/dt))    # compute number of euler steps to take
    t = t0+np.arange(steps+1)*dt        # array of time values at which to evaluate
    C = 0.*t                            # solution array
    C[0] = C0                           # set intial solution value
    
    # improved Euler method
    for i in range(steps):

        # predictor step
        predict_C = C[i]+dt*cf_ode_model_concentration(t[i],C[i],q_CO2[i],P[i], M0, a, b, P0, d, C0)

        # corrector step
        C[i+1] = C[i]+(cf_ode_model_concentration(t[i],C[i],q_CO2[i],P[i], M0, a, b, P0, d, C0)+cf_ode_model_concentration(t[i+1],predict_C,q_CO2[i+1],P[i+1],M0, a, b, P0, d, C0))*dt/2

    return C


def cf_solve_ode_pressure(t, t0, t1, dt, P0, q, dqdt, a, b, c):
    ''' Solve the Pressure ODE numerically by using improved euler method on ode_model_pressure()

        Parameters:
        -----------
        t : array-like
            Independent variable (time in years)
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        P0 : float
            Initial value of solution.
        q : array-like
            Injection/extraction rates.
        dqdt : array-like
            Injection/extraction rate derivatives.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        c : float
            Slow drainage parameter.

        Returns:
        --------
        P : array-like
            Dependent variable solution vector.

        Notes:
        ------
        The t parameter input is not used- it's purpose is soley to ensure compatibility with
        curve_fit by absorbing the indeprendent variable input from curve_fit.

    '''
    
    # initialise 
    steps = int(np.ceil((t1-t0)/dt))    # compute number of euler steps to take
    t = t0+np.arange(steps+1)*dt        # array of time values at which to evaluate
    P = 0.*t                            # solution array
    P[0] = P0                           # set intial solution value
    
    # improved Euler method
    for i in range(steps):

        # predictor step
        predict_P = P[i]+dt*cf_ode_model_pressure(P[i],q[i],dqdt[i],P0,a, b, c)

        # corrector step
        P[i+1] = P[i]+(cf_ode_model_pressure(P[i],q[i],dqdt[i],P0,a, b, c)+cf_ode_model_pressure(predict_P,q[i+1],dqdt[i+1],P0,a, b, c))*dt/2

    return P