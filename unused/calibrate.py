from matplotlib import pyplot as plt    # MATPLOTLIB is THE plotting module for Python
from sdlab_functions import*
from functions import *
from numpy.linalg import norm, solve
import numpy as np
from mpl_toolkits import mplot3d

def obj(theta,pressure,q_pressure,dqdt,f):
    [a,b,c] = theta
    numerical_x,numerical_y = solve_ode(f,1965,2018,0.1,6.17,[q_pressure, a, b, 6.17, c, dqdt])
    error = 0 
    for i in range(0,len(numerical_x)):
        error += ((numerical_y[i]-pressure[i])**2)**0.5
    return error

def obj_dir(obj, theta,pressure,q_pressure,dqdt,f, model=None):
    """ Compute a unit vector of objective function sensitivities, dS/dtheta.

        Parameters
        ----------
        obj: callable
            Objective function.
        theta: array-like
            Parameter vector at which dS/dtheta is evaluated.
        
        Returns
        -------
        s : array-like
            Unit vector of objective function derivatives.

    """
    # empty list to store components of objective function derivative 
    s = np.zeros(len(theta))
    
    # compute objective function at theta
    # **uncomment and complete the command below**
    s0 = obj(theta,pressure,q_pressure,dqdt,ode_model_pressure)


    # amount by which to increment parameter
    dtheta = 1.e-2
    
    # for each parameter
    for i in range(len(theta)):
        # basis vector in parameter direction 
        eps_i = np.zeros(len(theta))
        eps_i[i] = 1.
        
        # compute objective function at incremented parameter
        # **uncomment and complete the command below**
        si = obj(theta+(dtheta*eps_i),pressure,q_pressure,dqdt,ode_model_pressure)

        # compute objective function sensitivity
        # **uncomment and complete the command below**
        s[i] = (si - s0)/dtheta

    # return sensitivity vector
    return s,s0


# **this function is incomplete**
#					 ----------
def step(theta0, s, alpha):
    """ Compute parameter update by taking step in steepest descent direction.

        Parameters
        ----------
        theta0 : array-like
            Current parameter vector.
        s : array-like
            Step direction.
        alpha : float
            Step size.
        
        Returns
        -------
        theta1 : array-like
            Updated parameter vector.
    """
    # compute new parameter vector as sum of old vector and steepest descent step
    # **uncomment and complete the command below**
    theta1 = theta0 - (s*alpha)
    
    return theta1


# this function is complete
def line_search(obj, theta, s):
    """ Compute step length that minimizes objective function along the search direction.

        Parameters
        ----------
        obj : callable
            Objective function.
        theta : array-like
            Parameter vector at start of line search.
        s : array-like
            Search direction (objective function sensitivity vector).
    
        Returns
        -------
        alpha : float
            Step length.
    """
    # initial step size
    alpha = 0.
    # objective function at start of line search
    s0 = obj(theta)
    # anonymous function: evaluate objective function along line, parameter is a
    sa = lambda a: obj(theta-a*s)
    # compute initial Jacobian: is objective function increasing along search direction?
    j = (sa(.01)-s0)/0.01
    # iteration control
    N_max = 500
    N_it = 0
    # begin search
        # exit when (i) Jacobian very small (optimium step size found), or (ii) max iterations exceeded
    while abs(j) > 1.e-5 and N_it<N_max:
        # increment step size by Jacobian
        alpha += -j
        # compute new objective function
        si = sa(alpha)
        # compute new Jacobian
        j = (sa(alpha+0.01)-si)/0.01
        # increment
        N_it += 1
    # return step size
    return alpha

def ode_model_pressure(t, P, q, a, b, P0, c, dqdt):
    ''' Return the derivative dx/dt at time, t, for given parameters.

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
        >>> ode_model(0, 1, 2, 3, 4, 5, 6, 7)
        22

    '''
    
    dPdt=-a*q-b*(P-P0)-c*dqdt
    return dPdt




def main(): 

    # Set common timeline
    timeline = np.arange(1965,2018.1,0.1)

    # Find eqn for INJECTION;
    tw1,yw1 = np.genfromtxt('cs_c.txt',delimiter=',',skip_header=1).T 
    A = spline_coefficient_matrix(tw1) 
    b = spline_rhs(tw1,yw1)
    a_coeffs = solve(A,b)

    # Interpolate over common timeline.
    q_injection = spline_interpolate(timeline,tw1,a_coeffs)

    # Find eqn for PRESSURE;
    tw2,yw2 = np.genfromtxt('cs_p.txt',delimiter=',',skip_header=1).T 
    A = spline_coefficient_matrix(tw2) 
    b = spline_rhs(tw2,yw2)
    a_coeffs2 = solve(A,b)

    # Interpolate over common timeline.
    pressure = spline_interpolate(timeline,tw2,a_coeffs2)

    # Find eqn for PRODUCTION;
    tw3,yw3 = np.genfromtxt('cs_q.txt',delimiter=',',skip_header=1).T 
    A = spline_coefficient_matrix(tw3) 
    b = spline_rhs(tw3,yw3)
    a_coeffs3 = solve(A,b)

    # Interpolate over common timeline.
    q_production = spline_interpolate(timeline,tw3,a_coeffs3)

    q_pressure = q_production - q_injection

    dt = timeline[1]-timeline[0]
    dqdt = np.gradient(q_pressure, dt)
    theta = np.array([0.264,19.855,0.18])
    #c = 0.525

    s0,oj0 = obj_dir(obj,theta,pressure,q_pressure,dqdt,ode_model_pressure)
    
    alpha = 0.0001
    # update parameter estimate
    theta1 = step(theta, s0, alpha)
    
    #s1,oj1 = obj_dir(obj,theta1,c,pressure,q_pressure,dqdt,ode_model_pressure)
    
    theta_all = [theta]
    s_all = [s0]
    oj_all = [oj0]
    # iteration control
    N_max = 1500
    N_it = 0
    # begin steepest descent iterations
        # exit when max iterations exceeded
    while N_it < N_max:
        # uncomment line below to implement line search (TASK FIVE)
        #alpha = line_search(obj, theta_all[-1], s_all[-1])
        
        # update parameter vector 
        # **uncomment and complete the command below**
        theta_next = step(theta_all[N_it],s_all[N_it],alpha)
        theta_all.append(theta_next) 	# save parameter value for plotting
        
        # compute new direction for line search (thetas[-1]
        # **uncomment and complete the command below**
        s_next,oj_next = obj_dir(obj, theta_all[-1],pressure,q_pressure,dqdt,ode_model_pressure)
        s_all.append(s_next)
        oj_all.append(oj_next) 			# save search direction for plotting
        
        # compute magnitude of steepest descent direction for exit criteria
        N_it += 1
        # restart next iteration with values at end of previous iteration
        theta0 = 1.*theta_next
        s0 = 1.*s_next
    
    print('Optimum: ', round(theta_all[-1][0], 2), round(theta_all[-1][1], 2),round(theta_all[-1][2], 2))
    print('Number of iterations needed: ', N_it)
    print(min(oj_all))
    print(theta_all[np.argmin(oj_all)])

    # plot 4: compare against lab2_instructions.pdf, Figure 4 
    #plot_steps(obj, theta_all, s_all,pressure,q_pressure,dqdt)
    #exit function
    return
    
    
    """ A,B = np.meshgrid(a,b)
    Obj = objfunc(A,B,c,pressure,q_pressure,dqdt,ode_model_pressure)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(A, B, Obj, 50, cmap='binary')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('Obj') """

if __name__ == "__main__":
    main()
