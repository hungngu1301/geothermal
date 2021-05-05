
#begin stuff
from matplotlib import pyplot as plt    # MATPLOTLIB is THE plotting module for Python
from sdlab_functions import*
from functions import *
from numpy.linalg import norm, solve
import numpy as np
from scipy.interpolate import interp1d
tol = 1.e-6
def main():
    
    # Set common timeline
    timeline = np.arange(1968,2018.1,0.1)

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


    # Find eqn for CONCENTRAION;
    tw4,yw4 = np.genfromtxt('cs_cc.txt',delimiter=',',skip_header=1).T 
    A = spline_coefficient_matrix(tw4) 
    b = spline_rhs(tw4,yw4)
    a_coeffs4 = solve(A,b)

    # Interpolate over common timeline.
    concentration = spline_interpolate(timeline,tw4,a_coeffs4)

    q_pressure = q_production - q_injection
    dt = timeline[1]-timeline[0]
    dqdt = np.gradient(q_pressure, dt)
    a = 3e-3
    b = 2.26e-1
    c = 3.56e-5
    numerical_x,numerical_y=solve_ode(ode_model_pressure,1968,2018,0.1,6.17,[q_pressure, a, b, 6.17, c, dqdt])

    #Numerically solved Pressure
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax1.plot(numerical_x, numerical_y, 'r--', label='Modelled data')
    ax1.plot(numerical_x, pressure, 'k--', label='Interpolated data')
    ax1.set_ylim([3, 6])
    plt.show()

    
    #Analytically solved Pressure, benchmarking
    analytical_p=pressure_analytical(numerical_x,6.17,a,b,c)
    y=[150, 150]
    t=[1965, 2018]
    q_model=interp1d(t,y,kind='linear')
    dqdt_model = np.gradient(q_model(numerical_x), dt)
    t_benchmarking,P_benchmarking=solve_ode(ode_model_pressure,1968,2018,0.1,6.17,[q_model(numerical_x), a, b, 6.17, c, dqdt_model])
    f,ax1 = plt.subplots(nrows=1,ncols=1)

    
    #Compare with numerical
    ax1.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Time (yr)')
    ax1.plot(t_benchmarking, P_benchmarking, 'r--', label='Numerical')
    ax1.plot(t_benchmarking, analytical_p, 'b', label='Analytical')
    ax1.legend()
    ax1.set_title('Benchmarking for Pressure ODE')
    plt.show()

    #Step size
    h=np.linspace(0.01,5, num=200)
    P_step=[]
    for i in h:
        nx = int(np.ceil((2000-1968)/i))
        t = 1968+np.arange(nx+1)*i
        q_1=q_model(t)
        dqdt_model_1 = np.gradient(q_model(t), i)
        time,P=solve_ode(ode_model_pressure,1968,2000,i,6.17,[q_1, a, b, 6.17, c, dqdt_model_1])
        P_step.append(P[-1])
    
    #Convergence
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax1.set_ylabel('Pressure at 2000 (MPa)')
    ax1.set_xlabel('Step size h (yr)')
    ax1.plot(h,P_step, 'r', label='Modelled data')
    ax1.set_title('Convergence test with different time step')
    plt.show()

    #Concentration, all parameters are made up
    q=q_injection[0]
    M=0.28
    P=numerical_y[0]
    P0=numerical_y[0]
    d=0.3
    C0=0.03
    pars=[q,M,a,b,P,P0,d,C0]
    numerical_t,numerical_C=solve_ode_concentration(ode_model_concentration, 1968, 2018, 0.1, C0, q_injection, numerical_y, pars)

    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax1.plot(numerical_t, numerical_C, 'r--', label='Modelled data')
    ax1.plot(numerical_x, concentration, 'k--', label='Interpolated data')
    ax1.legend()
    plt.show()

    #Analytically solved concentration, benchmarking 
    #Assume q is constant at 50kg/s 
    q_injection_model=0.*np.array(q_injection)+50
    q=q_injection_model[0]

    tC_benchmarking,C_benchmarking=solve_ode_concentration(ode_model_concentration, 1968, 2018, 0.1, C0, q_injection_model, analytical_p, pars)
    analytical_C = concentration_analytical(numerical_x,q_injection_model,M,a,b,numerical_y,6.17,d,C0)
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    #Compare with numerical
    ax1.plot(tC_benchmarking, C_benchmarking, 'r--', label='Numerical')
    ax1.plot(numerical_x, analytical_C, 'b', label='Analytical')
    ax1.legend()
    ax1.set_ylabel('Concentration of CO2')
    ax1.set_xlabel('Time (yr)')
    ax1.set_title('Benchmarking for Concentration ODE')
    plt.show()

    #Step size
    h=np.linspace(0.001,5, num=200) 

    C_step=[]
    for i in h:
        nx = int(np.ceil((1990-1968)/i))
        t = 1968+np.arange(nx+1)*i
        q_1=q_model(t)
        dqdt_model_1 = np.gradient(q_model(t), i)
        q_injection_model=0.*np.array(t)+50
        time,P_con=solve_ode(ode_model_pressure,1968,1990,i,6.17,[q_1, a, b, 6.17, c, dqdt_model_1])
        time,C_1990=solve_ode_concentration(ode_model_concentration, 1968, 1990, i, C0, q_injection_model, P_con, pars)
        C_step.append(C_1990[-1])

    #Convergence
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax1.set_ylabel('Concentration at 1990')
    ax1.set_xlabel('Step size h (yr)')
    ax1.plot(h,C_step, 'r', label='Modelled data')
    ax1.set_title('Convergence test with different time step')
    plt.show()






    #-----------------------------------------------------------
    #Calibration for pressure, exit condition needs to be modified
    theta0=np.array([3e-3, 2.26e-1, 3.56e-5])
    s0=obj_dir(obj_pressure,theta0,pressure,[q_pressure,dqdt])
    alpha=0.00001

    theta_all=[theta0]
    s_all=[s0]

    N_max = 100
    N_it = 0
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax1.plot(numerical_x, pressure, 'ko', label='Interpolated data')
    numerical_x,numerical_y=solve_ode(ode_model_pressure,1968,2018,0.1,6.17,[q_pressure, theta0[0], theta0[1], 6.17, theta0[2], dqdt])    
    ax1.plot(numerical_x, numerical_y, 'r--', label='Orginal Modelled data')
    while N_it < N_max:
        # uncomment line below to implement line search (TASK FIVE)
        #alpha = line_search(obj, theta_all[-1], s_all[-1])
        
        # update parameter vector 
        # **uncomment and complete the command below**
        theta_next = step(theta_all[-1],s_all[-1],alpha)
        theta_all.append(theta_next) 	# save parameter value for plotting
        
        # compute new direction for line search (thetas[-1]
        # **uncomment and complete the command below**
        s_next = obj_dir(obj_pressure,theta_all[-1],pressure,[q_pressure,dqdt])
        s_all.append(s_next) 			# save search direction for plotting
        
        # compute magnitude of steepest descent direction for exit criteria
        N_it += 1
        # restart next iteration with values at end of previous iteration
        theta0 = 1.*theta_next
        s0 = 1.*s_next
    
    
    numerical_x,numerical_y=solve_ode(ode_model_pressure,1968,2018,0.1,6.17,[q_pressure, theta0[0], theta0[1], 6.17, theta0[2], dqdt])    
    ax1.plot(numerical_x, numerical_y, 'b', label='Calibrated Modelled data')
    ax1.legend()

    print('Optimum for pressure: ', round(theta_all[-1][0], 4), round(theta_all[-1][1], 4), round(theta_all[-1][2],4))
    print('Number of iterations needed: ', N_it)
    plt.show()

    f,ax1 = plt.subplots(nrows=1,ncols=1)
    numerical_x_P,numerical_y_P=solve_ode(ode_model_pressure,1968,2018,0.1,6.17,[q_pressure, theta0[0], theta0[1], 6.17, theta0[2], dqdt])    
    ax1.plot(numerical_x_P, numerical_y_P, 'r', label='Calibrated Model')
    ax1.plot(tw2,yw2,'bo', label='Given Data')
    ax1.set_title('Calibrated Pressure Model')
    ax1.set_ylabel('Presure (MPa)')
    ax1.set_xlabel('Time (yr)')
    ax1.set_ylim([3, 6])
    ax1.legend()
    plt.show()

    #Calibrate the concentration model 
    #Adjusting timeline for more accurate objective function 
    timeline_concentration=np.arange(1998,2018.1,0.1)
    q_injection_concentration = spline_interpolate(timeline_concentration,tw1,a_coeffs)
    concentration_2=spline_interpolate(timeline_concentration,tw4,a_coeffs4)
    pressure_2=spline_interpolate(timeline_concentration,tw2,a_coeffs2)
    numerical_y_2=numerical_y[np.where(numerical_x == 1998)[0][0]:np.where(numerical_x == 2018)[0][0]+1]

    a=0.0019741
    b=0.14734525
    theta0_C=np.array([0.254, 0.3094])
    s0=obj_dir(obj_concentration,theta0_C,concentration_2,[q_injection_concentration,pressure_2,a,b])
    alpha=0.0001

    theta_all=[theta0_C]
    s_all=[s0]

    ##Exit condition needs to be adjusted 
    N_max = 300
    N_it = 0
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    numerical_x_2,numerical_C=solve_ode_concentration(ode_model_concentration, 1998, 2018, 0.1, C0, q_injection_concentration, numerical_y_2, [q_injection_concentration[0],theta0_C[0],a,b,numerical_y_2[0],P0,theta0_C[1],C0])
    ax1.plot(numerical_x_2, concentration_2, 'ko', label='Interpolated data')   
    ax1.plot(numerical_x_2, numerical_C, 'r--', label='Orginal Modelled data')
    while N_it < N_max:
        # uncomment line below to implement line search (TASK FIVE)
        #alpha = line_search(obj, theta_all[-1], s_all[-1])
        
        # update parameter vector 
        # **uncomment and complete the command below**
        theta_next = step(theta_all[-1],s_all[-1],alpha)
        theta_all.append(theta_next) 	# save parameter value for plotting
        
        # compute new direction for line search (thetas[-1]
        # **uncomment and complete the command below**
        s_next = obj_dir(obj_concentration,theta_all[-1],concentration_2,[q_injection_concentration,pressure_2,a,b])
        s_all.append(s_next) 			# save search direction for plotting
        
        # compute magnitude of steepest descent direction for exit criteria
        N_it += 1
        # restart next iteration with values at end of previous iteration
        theta0_C = 1.*theta_next
        s0 = 1.*s_next

    
    numerical_x_2_C,numerical_C=solve_ode_concentration(ode_model_concentration, 1998, 2018, 0.1, C0, q_injection_concentration, numerical_y_2, [q_injection_concentration[0],theta0_C[0],a,b,numerical_y_2[0],P0,theta0_C[1],C0])    
    ax1.plot(numerical_x_2_C, numerical_C, 'b', label='Calibrated Modelled data')
    ax1.legend()

    print('Optimum for concentration: ', round(theta_all[-1][0], 6), round(theta_all[-1][1], 6))
    print('Number of iterations needed: ', N_it)
    plt.show()

    timeline_concentration_C=np.arange(1980,2018.1,0.1)
    q_injection_concentration_C = spline_interpolate(timeline_concentration_C,tw1,a_coeffs)
    concentration_2_C=spline_interpolate(timeline_concentration_C,tw4,a_coeffs4)
    pressure_2_C=spline_interpolate(timeline_concentration_C,tw2,a_coeffs2)
    numerical_y_2_P=numerical_y[np.where(numerical_x == 1980)[0][0]:np.where(numerical_x == 2018)[0][0]+1]

    f,ax1 = plt.subplots(nrows=1,ncols=1)
    numerical_x_C,numerical_y_C=solve_ode_concentration(ode_model_concentration, 1980, 2018, 0.1, C0, q_injection_concentration_C, numerical_y_2_P, [q_injection_concentration_C[0],theta0_C[0],a,b,numerical_y_2[0],P0,theta0_C[1],C0])    
    ax1.plot(numerical_x_C, numerical_y_C, 'r', label='Calibrated Model')
    ax1.plot(tw4,yw4,'bo', label='Given Data')
    ax1.set_title('Calibrated Concentration Model')
    ax1.set_ylabel('Concentration of CO2')
    ax1.set_xlabel('Time (yr)')
    ax1.legend()
    plt.show()


    y=[-50, -50]
    t=[1965, 2050]
    q_model=interp1d(t,y,kind='linear')
    timeline_concentration=np.arange(2018,2050.1,0.1)
    q_1=q_model(timeline_concentration)
    dqdt_model_1 = np.gradient(q_1, 0.1)
    numerical_x_2,numerical_y_2=solve_ode(ode_model_pressure,2018,2050,0.1,numerical_y_P[-1],[q_1, theta0[0], theta0[1], 6.17, theta0[2], dqdt_model_1])

    y=[50, 50] #2q
    t=[1965, 2050]
    q_model=interp1d(t,y,kind='linear')
    timeline_concentration=np.arange(2018,2050.1,0.1)
    q_1=q_model(timeline_concentration)
    dqdt_model_1 = np.gradient(q_1, 0.1)
    numerical_x_3,numerical_y_3=solve_ode(ode_model_pressure,2018,2050,0.1,numerical_y_P[-1],[q_1, theta0[0], theta0[1], 6.17, theta0[2], dqdt_model_1])

    y=[0, 0] #3q
    t=[1965, 2050]
    q_model=interp1d(t,y,kind='linear')
    timeline_concentration=np.arange(2018,2050.1,0.1)
    q_1=q_model(timeline_concentration)
    dqdt_model_1 = np.gradient(q_1, 0.1)
    numerical_x_4,numerical_y_4=solve_ode(ode_model_pressure,2018,2050,0.1,numerical_y_P[-1],[q_1, theta0[0], theta0[1], 6.17, theta0[2], dqdt_model_1])

    n=len(numerical_x)
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax1.set_ylabel('Pressure')
    ax1.set_xlabel('Time (yr)')
    ax1.plot(numerical_x, numerical_y, 'b', label='Current')
    ax1.plot(numerical_x_2,numerical_y_2, 'r--', label='Quadruple injection')
    ax1.plot(numerical_x_3,numerical_y_3, 'b--', label='Double injection')
    ax1.plot(numerical_x_4,numerical_y_4, 'k--', label='Triple injection (q=0)')
    ax1.set_title('Prediction for Pressure at Ohaaki')
    ax1.legend()
    plt.show()


    y=[200, 200]
    t=[1965, 2050]
    q_model=interp1d(t,y,kind='linear')
    timeline_concentration=np.arange(2018,2050.1,0.1)
    q_1=q_model(timeline_concentration)
    numerical_x_2,numerical_C_2=solve_ode_concentration(ode_model_concentration, 2018, 2050, 0.1, numerical_C[-1], q_1, numerical_y_2, [200,theta0_C[0],a,b,numerical_y_2[0],P0,theta0_C[1],C0])

    y=[100, 100] #2q
    t=[1965, 2050]
    q_model=interp1d(t,y,kind='linear')
    timeline_concentration=np.arange(2018,2050.1,0.1)
    q_1=q_model(timeline_concentration)
    numerical_x_3,numerical_C_3=solve_ode_concentration(ode_model_concentration, 2018, 2050, 0.1, numerical_C[-1], q_1, numerical_y_3, [100,theta0_C[0],a,b,numerical_y_2[0],P0,theta0_C[1],C0])

    y=[150, 150] #3q
    t=[1965, 2050]
    q_model=interp1d(t,y,kind='linear')
    timeline_concentration=np.arange(2018,2050.1,0.1)
    q_1=q_model(timeline_concentration)
    numerical_x_4,numerical_C_4=solve_ode_concentration(ode_model_concentration, 2018, 2050, 0.1, numerical_C[-1], q_1, numerical_y_4, [150,theta0_C[0],a,b,numerical_y_2[0],P0,theta0_C[1],C0])

    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax1.set_ylabel('Concentration')
    ax1.set_xlabel('Time (yr)')
    ax1.plot(numerical_x_2_C, numerical_C, 'b', label='Current')
    ax1.plot(numerical_x_2,numerical_C_2, 'r--', label='Quadruple injection')
    ax1.plot(numerical_x_3,numerical_C_3, 'b--', label='Double injection')
    ax1.plot(numerical_x_4,numerical_C_4, 'k--', label='Triple injection')
    ax1.set_title('Prediction for Concentration of CO2 at Ohaaki')
    ax1.legend()
    plt.show()


    #-----------------------------------------------------------------
    #Posterior
    theta0=[0.0019741, 0.14734525, 0.00098846]
    q_set_P=[]
    dqdt_model_set=[]
    timeline_concentration=np.arange(2018,2040.1,0.1)
    t=[2018, 2040]
    f,ax =plt.subplots(1,1)
    #ax.plot(numerical_x, numerical_y, 'b', label='Current')
    rate=[-50,0,50]
    s=['b-','r-','k-']
    for i in range(len(rate)):
        y=[rate[i], rate[i]]
        q_model=interp1d(t,y,kind='linear')
        q_set_P.append(q_model(timeline_concentration))
        dqdt_model_set.append(np.gradient(q_set_P[-1], 0.1)) 
        t_pos,P_post=solve_ode(ode_model_pressure,1968,2018,0.1,6.17,[q_pressure, theta0[0], theta0[1], 6.17, theta0[2], dqdt])
        a,b,c,posterior = grid_search_pressure(P_post,q_pressure,dqdt,6)
        N=40
        samples = construct_samples_pressure(a, b, c, posterior, N)
        P_set=model_ensemble_pressure(samples,np.append(q_pressure,q_set_P[i]),np.append(dqdt,dqdt_model_set[i]),6)
        for j in range(len(P_set)):
            ax.plot(np.append(t_pos,timeline_concentration[1:]),P_set[j],s[i], label='q={}'.format(rate[i]),lw=0.5,alpha=0.4)
    

    plt.show()

    #--------------------------------------------------------------------
    q_set_C=[]
    timeline_concentration=np.arange(2018,2040.1,0.1)
    t=[2018, 2040]
    f,ax =plt.subplots(1,1)
    ax.plot(numerical_x_2_C, numerical_C, 'b', label='Current')
    rate=[200,150,100]
    for i in range(len(rate)):
        y=[rate[i], rate[i]]
        q_model=interp1d(t,y,kind='linear')
        q_set_C.append(q_model(timeline_concentration))
        t_post,P_post=solve_ode(ode_model_pressure,2018,2040,0.1,numerical_y_P[-1],[q_set_P[i], theta0[0], theta0[1], 6.17, theta0[2], dqdt_model_set[i]])
        t_post,C_post=solve_ode_concentration(ode_model_concentration, 2018, 2040, 0.1, numerical_C[-1], q_set_C[i], P_post, [rate[i],theta0_C[0],theta0[0],theta0[1],P_post[0],P0,theta0_C[1],C0])
        M_pos,d_pos,posterior = grid_search_concentration(C_post,P_post,q_set_C[i])
        N=20
        samples = construct_samples_concentration(M_pos, d_pos, posterior, N)
        C_set=model_ensemble_concentration(samples,q_set_C[i],P_post)
        for j in range(len(C_set)):
            ax.plot(t_post,C_set[j],s[i], label='q_inj={}'.format(rate[i]),lw=0.5,alpha=0.4)

    plt.show()



if __name__ == "__main__":
    main()
