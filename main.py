from matplotlib import pyplot as plt    
from functions import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def main():
    # ---------------
    # | Interpolate |
    # ---------------

    # Set common timeline
    timeline = np.arange(1968,2018.1,0.1)

    #  Find eqn for INJECTION;
    # -------------------------
    
    tw1,yw1 = np.genfromtxt('cs_c.txt',delimiter=',',skip_header=1).T   #Read in Injection data 
    time_inj=np.arange(1999,2018.1,0.1)     #Create timeline over which data is to be interpolated
    inj_before=np.arange(1968,1999,0.1)*0   #Create an array of 0s to be the points before injection starts
    model=interp1d(tw1,yw1,kind='cubic')    #Create a cubic spline matrix set for interpolation
    
    # Interpolate over common timeline
    q_injection_plot=model(time_inj)    #Get interpolated values for desired timeline 
    q_injection = np.append(inj_before, q_injection_plot)   #Append interpolated values with array of 0s to represent full common timeline


    #  Find eqn for PRESSURE
    # ------------------------

    tw2,yw2 = np.genfromtxt('cs_p.txt',delimiter=',',skip_header=1).T   #Read in Pressure data 
    model=interp1d(tw2,yw2,kind='cubic')    #Create a cubic spline matrix set for interpolation

    # Interpolate over common timeline
    pressure = model(timeline)  #Get interpolated values for desired timeline 


    #  Find eqn for PRODUCTION
    # --------------------------

    tw3,yw3 = np.genfromtxt('cs_q.txt',delimiter=',',skip_header=1).T   #Read in Production data 
    model=interp1d(tw3,yw3,kind='cubic')    #Create a cubic spline matrix set for interpolation

    # Interpolate over common timeline
    q_production = model(timeline)   #Get interpolated values for desired timeline 


    #  Find eqn for CONCENTRAION
    # ----------------------------
   
    con_before=np.arange(1968,1980,0.1)*0+0.03
    time_con=np.arange(1980,2018.1,0.1)
    tw4,yw4 = np.genfromtxt('cs_cc.txt',delimiter=',',skip_header=1).T  #Read in Concentration data 
    model=interp1d(tw4,yw4,kind='cubic')    #Create a cubic spline matrix set for interpolation

    # Interpolate over common timeline
    concentration = np.append(con_before,model(time_con))   #Get interpolated values and add initial concentration values for the time before production 


    # ---------
    # | Given |
    # ---------

    #  Injection vs Concentration
    # ----------------------------

    f,ax1 = plt.subplots(nrows=1,ncols=1)   #Create a figure
    ax2 = ax1.twinx()   #Create a twin y-axis
    ax1.set_title('Comparision of CO2 injection rate & CO2 weight fraction in Reservoir',y=1.08)    #Set title for figure
    ax1.set_ylabel('CO2 Injection Rates (kg/s)')    #Labelling the y-axes
    ax2.set_ylabel('CO2 weight fraction in Reservior')
    ax1.set_xlabel('Year')  #Labelling the x-axis
    ax2.plot(tw4,yw4,'r',label='CO2 concentration') #Plotting interpolated models on with respect to appropriate y-axis
    ax1.plot(np.append(np.arange(1968,1998.6,0.1),tw1),np.append(np.arange(1968,1998.6,0.1)*0,yw1),'b',label='CO2 Injection') 
    ax1.legend(bbox_to_anchor=(0., 1.02, 0.5, .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)     #Creating legends for plots 
    ax2.legend(loc=2) 
    plt.savefig('Injection_VS_concentration.png',dpi=300)   #Saving the figure


    #  Extraction vs Pressure 
    # ------------------------

    f,ax1 = plt.subplots(nrows=1,ncols=1)    #Create a figure
    ax2 = ax1.twinx()   #Create a twin y-axis
    ax1.set_title('Comparision of extraction/injection rate & Pressure in Reservoir',y=1.08)    #Set title for figure
    ax1.set_ylabel('Mass Rates (kg/s)')     #Labelling the y-axes 
    ax2.set_ylabel('Pressure (MPa)')
    ax1.set_xlabel('Year')  #Labelling the x-axis
    ax2.plot(tw2,yw2,'k--',label='Pressure')       #Plotting interpolated models on with respect to appropriate y-axis
    ax1.plot(np.append(np.arange(1968,1998.6,0.1),tw1),np.append(np.arange(1968,1998.6,0.1)*0,yw1),'r',label='Injection') 
    ax1.plot(tw3,yw3,'b',label='Extraction')
    ax1.legend(bbox_to_anchor=(0., 1.02, 0.5, .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)     #Creating legends for plots
    ax2.legend(loc=2) 
    plt.savefig('MassRate_VS_Pressure.png',dpi=300)     #Saving the figure




    # ----------------
    # | Benchmarking |
    # ----------------

    #Setting constants and parameters 
    q_pressure = q_production - q_injection
    dt = timeline[1]-timeline[0]
    dqdt = np.gradient(q_pressure, dt)
    a = 3e-3
    b = 2.26e-1
    c = 3.56e-5

    #Solving Pressure model numerically 
    numerical_x,numerical_y=solve_ode(ode_model_pressure,1968,2018,0.1,6.17,[q_pressure, a, b, 6.17, c, dqdt])

    #Analytically solved Pressure, benchmarking
    analytical_p=pressure_analytical(timeline,6.17,a,b,c)
    y=[150, 150]
    t=[1968, 2018]
    q_model=interp1d(t,y,kind='linear')
    dqdt_model = np.gradient(q_model(timeline), dt)
    t_benchmarking,P_benchmarking=solve_ode(ode_model_pressure,1968,2018,0.1,6.17,[q_model(timeline), a, b, 6.17, c, dqdt_model])
    
    #Compare with numerical
    f,ax1 = plt.subplots(nrows=1,ncols=1)   #Create a figure
    ax1.set_ylabel('Pressure (MPa)')     #Labelling the y-axis 
    ax1.set_xlabel('Time (yr)')      #Labelling the x-axis 
    ax1.plot(t_benchmarking, P_benchmarking, 'r', label='Numerical')    #Plotting Numerical Solution 
    ax1.plot(t_benchmarking, analytical_p, 'bx', label='Analytical', alpha=0.2)     #Plotting Analytical Solution
    ax1.legend()    #Displaying Legend for Figure 
    ax1.set_title('Benchmarking for Pressure ODE')  #Setting Title for figure 
    plt.savefig('Pressure_Benchmarking.png')    #Saving the figure 

    #Step size test
    h=np.linspace(0.01,5, num=200) #Creating an array of step sizes to test
    P_step=[] #Blank array to store test values
    for i in h:
        nx = int(np.ceil((2000-1968)/i))
        t = 1968+np.arange(nx+1)*i
        q_1=q_model(t)
        dqdt_model_1 = np.gradient(q_model(t), i)
        time,P=solve_ode(ode_model_pressure,1968,2000,i,6.17,[q_1, a, b, 6.17, c, dqdt_model_1])
        P_step.append(P[-1])
    
    #Convergence
    f,ax1 = plt.subplots(nrows=1,ncols=1) #Create a figure
    ax1.set_ylabel('Pressure at 2000 (MPa)') #Labelling the y-axis
    ax1.set_xlabel('Step size h (yr)') #Labelling the x-axis 
    ax1.plot(h,P_step, 'r', label='Modelled data') #Plot values against step sizes
    ax1.set_title('Convergence test with different time step') #Setting Title for figure 
    plt.savefig('Pressure_Convergence.png')  #Saving the figure

    #Concentration, all parameters are made up
    q=q_injection[0]
    M=8000
    P=numerical_y[0]
    P0=numerical_y[0]
    d=0.3
    C0=0.03
    pars=[q,M,a,b,P,P0,d,C0]

    #Analytically solved concentration, benchmarking 
    #Assume q is constant at 50kg/s 
    q_injection_model=0.*np.array(q_injection)+50
    q=q_injection_model[0]
    tC_benchmarking,C_benchmarking=solve_ode_concentration(ode_model_concentration, 1968, 2018, 0.1, C0, q_injection_model, analytical_p, pars)
    analytical_C = concentration_analytical(numerical_x,q_injection_model,M,a,b,analytical_p,6.17,d,C0)

    #Compare with numerical
    f,ax1 = plt.subplots(nrows=1,ncols=1)   #Create a figure
    ax1.plot(tC_benchmarking, C_benchmarking, 'r', label='Numerical')   #Plotting Numerical Solution
    ax1.plot(numerical_x, analytical_C, 'bx', label='Analytical', alpha=0.2)     #Plotting Analytical Solution
    ax1.legend()    #Displaying Legend for Figure 
    ax1.set_ylabel('Concentration of CO2')  #Labelling the y-axis
    ax1.set_xlabel('Time (yr)')     #Labelling the x-axis 
    ax1.set_title('Benchmarking for Concentration ODE')     #Setting Title for figure  
    plt.savefig('Concentration_Benchmarking.png')   #Saving the figure

    #Step size test
    h=np.linspace(0.001,5, num=200) #Creating an array of step sizes to test
    C_step=[]   #Blank array to store test values
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
    f,ax1 = plt.subplots(nrows=1,ncols=1)   #Create a figure
    ax1.set_ylabel('Concentration at 1990')  #Labelling the y-axis
    ax1.set_xlabel('Step size h (yr)')   #Labelling the x-axis 
    ax1.plot(h,C_step, 'r', label='Modelled data') #Plot values against step sizes
    ax1.set_title('Convergence test with different time step') #Setting Title for figure  
    plt.savefig('Concentration_Convergence.png') #Saving the figure


    # ---------------
    # | Calibration |
    # ---------------

    #  curve_fit Calibration for PRESSURE ODE
    # ----------------------------------------

    # initial guess for parameters based on literature (Fradkin et al 1981)
    P_guess = np.array([0.01, 0.39, 1.3])

    # use curve fit to calibrate parameters a, b and c
    abc_calibrated, P_covariance = curve_fit(lambda cf_t,a_parm,b_parm,c_parm: cf_solve_ode_pressure(cf_t, timeline[0], timeline[-1], dt, pressure[0], q_pressure, dqdt, a_parm, b_parm, c_parm), timeline, pressure, p0=P_guess)
    
    #print(abc_calibrated) # uncomment to check parameter values
    #print(P_covariance) # uncomment to print covariance matrix

    # plot pressure model against pressure data
    f,ax1 = plt.subplots(nrows=1,ncols=1)   #Create a figure
    ax1.plot(tw2,yw2, 'ro', label='Given data') # Plot the given data for pressure
    ax1.plot(timeline, cf_solve_ode_pressure(timeline, timeline[0], timeline[-1], dt, pressure[0], q_pressure, dqdt, abc_calibrated[0], abc_calibrated[1], abc_calibrated[2]), 'b', label='Model')  #Plot calibrated model
    ax1.set_title('Pressure ODE: data vs. model comparison')    #Set Figure title 
    ax1.set_ylabel('Pressure (MPa)')    #Labelling y-axis
    ax1.set_xlabel('Time (yr)') #Labelling x-axis
    ax1.legend()    #Displaying Legend 
    plt.savefig('Pressure_Calibration.png')  #Saving the figure 


    #  curve_fit calibration CONCENTRATION ODE
    # -----------------------------------------

    # use pressure model to get solution for pressure values
    P_model = cf_solve_ode_pressure(timeline, timeline[0], timeline[-1], dt, pressure[0], q_pressure, dqdt, abc_calibrated[0], abc_calibrated[1], abc_calibrated[2])

    # initial guess for d and M0 (educated-ish guess)
    C_guess = np.array([3*10**7, 0.3]) 

    # use curve fit to calibrate parameters M0 and d
    M0d_calibrated, C_covariance = curve_fit(lambda cf_t,M0_parm,d_parm: cf_solve_ode_concentration(cf_t, timeline[0], timeline[-1], dt, concentration[0], q_injection, P_model, M0_parm, abc_calibrated[0], abc_calibrated[1], pressure[0], d_parm), timeline, concentration, p0=C_guess)
    
    #print(M0d_calibrated) # uncomment to check parameter values
    #print(M0d_covariance) # uncomment to print covariance matrix

    # plot pressure model against Concentration data
    f,ax1 = plt.subplots(nrows=1,ncols=1)   #Create a figure
    ax1.plot(tw4, yw4, 'ro', label='Given data')    # Plot the given data for concentration 
    ax1.plot(timeline, cf_solve_ode_concentration(timeline, timeline[0], timeline[-1], dt, concentration[0], q_injection, P_model, M0d_calibrated[0], abc_calibrated[0], abc_calibrated[1], pressure[0], M0d_calibrated[1]), 'b', label='Model')    #Plot calibrated model
    ax1.set_title('CO2 Concentration ODE: data vs. model comparison')
    ax1.set_ylabel('Concentration') #Labelling y-axis
    ax1.set_xlabel('Time (yr)') #Labelling x-axis
    ax1.legend()   #Displaying Legend  
    plt.savefig('Concentration_Calibration.png')    #Saving the figure

    # using calibrated parameters to find pressure and concentration values according to model:
    #Pressure
    thetaP=[abc_calibrated[0], abc_calibrated[1], abc_calibrated[2]] # get calibrated a, b and c values
    time,P_calibrated=solve_ode(ode_model_pressure,1968,2018,0.1,6.17,[q_pressure, thetaP[0], thetaP[1], 6.17, thetaP[2], dqdt]) # solve for pressure

    #Concentration
    thetaC=[M0d_calibrated[0], M0d_calibrated[1]] # get calibrated M0 and d values
    P_concentration=P_calibrated[np.where(time == 1980)[0][0]:np.where(time == 2018)[0][0]+1] # modify pressure values to fit concentration timeline
    q_injection=q_injection[np.where(time == 1980)[0][0]:np.where(time == 2018)[0][0]+1] # modify injection values to fit concentration timline
    time_C,C_calibrated=solve_ode_concentration(ode_model_concentration, 1980, 2018, 0.1, C0, q_injection, P_concentration, [q_injection[0],thetaC[0],thetaP[0],thetaP[1],P_calibrated[-1],P0,thetaC[1],C0]) # solve for concentration
    
    # ----------
    # | Misfit |
    # ----------

    #Pre-allocate list to store misfit
    misfitP_time=[]
    misfitP=[]
    #loop through the data
    for i in range(1,len(tw2)-1):
        #find the index where calibrated data matches with given data 
        j=np.where(np.round(time,decimals=1) == np.round(tw2[i],decimals=1))[0][0]

        #misfit
        misfitP.append(P_calibrated[j]-yw2[i])

        #store
        misfitP_time.append(time[j])

    #Graph and save plot
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax1.set_ylabel('Misfit (MPa)')
    ax1.set_xlabel('Time (yr)')
    ax1.plot(misfitP_time, misfitP, 'rx', label='Misfit')
    ax1.plot(np.linspace(1968,2018),np.linspace(1968,2018)*0, 'k--', alpha=0.5)
    ax1.set_title('Misfit of calibrated pressure model')
    plt.savefig('Pressure_Misfit.png')

    #Pre-allocate list to store misfit
    misfitC_time=[]
    misfitC=[]

    for i in range(1,len(tw4)-1):
        #find the index where calibrated data matches with given data 
        j=np.where(np.round(time_C,decimals=1) == np.round(tw4[i],decimals=1))[0][0]

        #misfit
        misfitC.append(C_calibrated[j]-yw4[i])

        #store
        misfitC_time.append(time_C[j])

    #Graph and save plot
    f,ax1 = plt.subplots(nrows=1,ncols=1)   #Create a figure
    ax1.set_ylabel('Misfit')    #Labelling the y-axis
    ax1.set_xlabel('Time (yr)')  #Labelling the x-axis
    ax1.plot(misfitC_time, misfitC, 'rx', label='Misfit') 
    ax1.plot(np.linspace(1980,2018),np.linspace(1980,2018)*0, 'k--', alpha=0.5)
    ax1.set_title('Misfit of calibrated concentration model') #Setting Title for figure  
    plt.savefig('Concentration_Misfit.png')    #Saving the figure
    
    # --------------
    # | Prediction |
    # --------------

    #  Predictions for pressure
    # --------------------------
    
    #Quadrupling Injection
    y=[-50, -50]
    t=[2018, 2050] #prediction time 
    #Interpolate
    q_model=interp1d(t,y,kind='linear')
    timeline_prediction=np.arange(2018,2050.1,0.1)
    q_1=q_model(timeline_prediction)
    dqdt_model_1 = np.gradient(q_1, 0.1)
    t_prediction,P_prediction_1=solve_ode(ode_model_pressure,2018,2050,0.1,P_calibrated[-1],[q_1, thetaP[0], thetaP[1], 6.17, thetaP[2], dqdt_model_1])

    #Triple, and permanent stop
    y=[0, 0] 
    q_model=interp1d(t,y,kind='linear')
    q_1=q_model(timeline_prediction)
    dqdt_model_1 = np.gradient(q_1, 0.1)
    t_prediction,P_prediction_2=solve_ode(ode_model_pressure,2018,2050,0.1,P_calibrated[-1],[q_1, thetaP[0], thetaP[1], 6.17, thetaP[2], dqdt_model_1])

    #Doubling Injection
    y=[50, 50] 
    q_model=interp1d(t,y,kind='linear')
    q_1=q_model(timeline_prediction)
    dqdt_model_1 = np.gradient(q_1, 0.1)
    t_prediction,P_prediction_3=solve_ode(ode_model_pressure,2018,2050,0.1,P_calibrated[-1],[q_1, thetaP[0], thetaP[1], 6.17, thetaP[2], dqdt_model_1])

    #Unchanged Injection
    y=[100, 100] 
    q_model=interp1d(t,y,kind='linear')
    q_1=q_model(timeline_prediction)
    dqdt_model_1 = np.gradient(q_1, 0.1)
    t_prediction,P_prediction_4=solve_ode(ode_model_pressure,2018,2050,0.1,P_calibrated[-1],[q_1, thetaP[0], thetaP[1], 6.17, thetaP[2], dqdt_model_1])

    #Graph
    f,ax1 = plt.subplots(nrows=1,ncols=1)   #Create a figure
    ax1.set_ylabel('Pressure (MPa)')    #Labelling y-axis
    ax1.set_xlabel('Time (yr)')     #Labelling x-axis
    ax1.plot(time, P_calibrated, 'b', label='Best fit model')
    ax1.plot(tw2, yw2, 'ro', label='Given data') 
    ax1.plot(np.linspace(1968,2050),np.linspace(1968,2050)*0+6.17, 'ko', alpha=0.2)
    ax1.plot(t_prediction,P_prediction_1, 'r--', label='Quadruple q=-50 kg/s')
    ax1.plot(t_prediction,P_prediction_2, 'k--', label='Triple/perma stop q=0 kg/s')
    ax1.plot(t_prediction,P_prediction_3, 'y--', label='Double q=50 kg/s')
    ax1.plot(t_prediction,P_prediction_4, 'b--', label='Unchange q=100 kg/s')
    ax1.set_title('Prediction for Pressure at Ohaaki')  #Setting Title for figure  
    ax1.legend(loc=4)   #Displaying Legend for Figure 
    plt.savefig('Pressure_Predictions.png') #Saving Figure


    #  Predictions for Concentration
    # ------------------------------

    #Quadrupling Injection
    y=[200, 200]
    q_model=interp1d(t,y,kind='linear')
    q_1=q_model(timeline_prediction)
    t_prediction,C_prediction_1=solve_ode_concentration(ode_model_concentration, 2018, 2050, 0.1, C_calibrated[-1], q_1, P_prediction_1, [200,thetaC[0],thetaP[0],thetaP[1],P_calibrated[-1],P0,thetaC[1],C0])
    
    #Tripling Injection
    y=[150, 150] 
    q_model=interp1d(t,y,kind='linear')
    q_1=q_model(timeline_prediction)
    t_prediction,C_prediction_2=solve_ode_concentration(ode_model_concentration, 2018, 2050, 0.1, C_calibrated[-1], q_1, P_prediction_2, [150,thetaC[0],thetaP[0],thetaP[1],P_calibrated[0],P0,thetaC[1],C0])
    
    #Doubling Injection
    y=[100, 100] 
    q_model=interp1d(t,y,kind='linear')
    q_1=q_model(timeline_prediction)
    t_prediction,C_prediction_3=solve_ode_concentration(ode_model_concentration, 2018, 2050, 0.1, C_calibrated[-1], q_1, P_prediction_3, [100,thetaC[0],thetaP[0],thetaP[1],P_calibrated[0],P0,thetaC[1],C0])

    #Unchanged Injection 
    y=[50, 50] 
    q_model=interp1d(t,y,kind='linear')
    q_1=q_model(timeline_prediction)
    t_prediction,C_prediction_4=solve_ode_concentration(ode_model_concentration, 2018, 2050, 0.1, C_calibrated[-1], q_1, P_prediction_4, [50,thetaC[0],thetaP[0],thetaP[1],P_calibrated[0],P0,thetaC[1],C0])
    
    #Permanent stop to Injection
    y=[0, 0] 
    q_model=interp1d(t,y,kind='linear')
    q_1=q_model(timeline_prediction)
    t_prediction,C_prediction_5=solve_ode_concentration(ode_model_concentration, 2018, 2050, 0.1, C_calibrated[-1], q_1, P_prediction_2, [0,thetaC[0],thetaP[0],thetaP[1],P_calibrated[0],P0,thetaC[1],C0])
    
    #Graph
    f,ax1 = plt.subplots(nrows=1,ncols=1)    #Create a figure
    ax1.set_ylabel('Concentration')     #Labelling y-axis
    ax1.set_xlabel('Time (yr)')     #Labelling x-axis
    ax1.plot(time_C, C_calibrated, 'b', label='Best-fit model')
    ax1.plot(tw4, yw4, 'ro', label='Given data')
    ax1.plot(np.linspace(1980,2050),np.linspace(1980,2050)*0+0.1, 'ko', alpha=0.2)
    ax1.plot(t_prediction,C_prediction_1, 'r--', label='Quadruple q=200 kg/s')
    ax1.plot(t_prediction,C_prediction_2, 'k--', label='Triple q=150 kg/s')
    ax1.plot(t_prediction,C_prediction_3, 'y--', label='Double q=100 kg/s')
    ax1.plot(t_prediction,C_prediction_4, 'b--', label='Unchanged q=50 kg/s')
    ax1.plot(t_prediction,C_prediction_5, 'rx', label='Permanent stop q=0')
    ax1.set_title('Prediction for Concentration of CO2 at Ohaaki')  #Setting Title for figure 
    ax1.legend()    #Displaying Legend for Figure
    plt.savefig('Concentration_Predictions.png')    #Saving Figure


    # --------------------------
    # | Uncertainty: Posterior | 
    # --------------------------

    #Pre-allocate list to store data 
    q_posterior=[] #store for latter used in concentration
    dqdt_posterior=[] #store for latter used in concentration
    P_upperbound=[] #5,95
    P_lowerbound=[]

    #Create a plot
    f,ax =plt.subplots(1,1)

    #List of rates q=q_ext-q_inj
    rate_net=[-50,0,50,100,0]

    #Pre-allocate colours 
    s=['r-','k-','y-','b-','ro'] #colour
    for i in range(len(rate_net)):
        #Interpolate constant rate 
        y=[rate_net[i], rate_net[i]]
        q_model=interp1d(t,y,kind='linear')

        #store
        q_posterior.append(q_model(timeline_prediction))
        dqdt_posterior.append(np.gradient(q_posterior[i], 0.1))

        #solve to get t_Pos
        t_pos,P_post=solve_ode(ode_model_pressure,1968,2018,0.1,6.17,[q_pressure, thetaP[0], thetaP[1], 6.17, thetaP[2], dqdt])

        #Create N number of samples
        N=1000
        samples = construct_samples_pressure(thetaP, P_covariance, N)

        #Solve model with all samples 
        P_set=model_ensemble_pressure(samples,np.append(q_pressure,q_posterior[i]),np.append(dqdt,dqdt_posterior[i]),pressure[0])

        #record steady state solution
        P_last=[]
        for j in range(len(P_set)):
            P_last.append(P_set[j][-1])
            #Graph
            if (i==4) is False:
                ax.plot(np.append(t_pos,timeline_prediction[1:]),P_set[j],s[i], label='q={}'.format(rate_net[i]),lw=0.2)
        
        #store
        P_upperbound.append(np.max(P_last))
        P_lowerbound.append(np.min(P_last))
    

    #print(P_upperbound) # uncomment to print upperbound of pressure prediction
    #print(P_lowerbound) # uncomment to print lowerbound of pressure prediction

    #Given data with error bars
    ax.errorbar(tw2,yw2,yerr=0.2,fmt='ro', label='data')

    #Modify the legend ref: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)

    plt.legend(handle_list, label_list)

    #Pre-exploitation pressure 
    ax.plot(np.linspace(1968,2050),np.linspace(1968,2050)*0+6.17, 'ko', alpha=0.2)

    #Title and labels 
    ax.set_title('Prediction for Pressure at Ohaaki with Uncertainty')
    ax.set_ylabel('Pressure (MPa)')
    ax.set_xlabel('Time (yr)')
    plt.savefig('Pressure_Predictions_Uncertainty.png')

    #concentration 
    #pre-allocation
    C_upperbound=[]
    C_lowerbound=[]

    #create a plot
    f,ax =plt.subplots(1,1)

    #calibrated model
    ax.plot(time_C, C_calibrated, 'k-', label='Current')

    #rate of injection
    rate=[200,150,100,50,0]

    for i in range(len(rate)):
        #interpolate constant rate over time t
        y=[rate[i], rate[i]]
        q_model=interp1d(t,y,kind='linear')
        q_inj_posterior=q_model(timeline_prediction)

        #Solve for corresponding pressure
        t_useless,P_prediction_posterior=solve_ode(ode_model_pressure,2018,2050,0.1,P_calibrated[-1],[q_posterior[i], thetaP[0], thetaP[1], 6.17, thetaP[2], dqdt_posterior[i]])

        #Create N number of samples 
        N=1000
        samples = construct_samples_concentration(thetaC,C_covariance , N)

        #Solve the model with N samples
        C_set=model_ensemble_concentration(samples,np.append(q_injection,q_inj_posterior),np.append(P_concentration,P_prediction_posterior)) #from 1980 to 2050
        
        #Record steady solution
        C_last=[]
        for j in range(len(C_set)):
            C_last.append(C_set[j][-1])
            #Graph
            if i==4:
                ax.plot(timeline_prediction[1:],C_set[j][np.where(np.round(np.arange(1980,2050.1,0.1),decimals=1)==timeline_prediction[1])[0][0]:],s[i], label='q_inj={}'.format(rate[i]),lw=0.2)
            else:
                ax.plot(np.append(time_C,timeline_prediction[1:]),C_set[j],s[i], label='q_inj={}'.format(rate[i]),lw=0.2)
        
        #Store upper and lowerbound
        C_upperbound.append(np.max(C_last))
        C_lowerbound.append(np.min(C_last))

    #print(C_upperbound) #uncomment to print upperbounds for concentration predictions
    #print(C_lowerbound) #uncomment to print lowerbounds for concentration predictions
    
    #Given data with error bar 
    ax.errorbar(tw4,yw4,yerr=0.0025,fmt='ro', label='Data')

    #Modify the legend
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    plt.legend(handle_list, label_list)

    #10% line
    ax.plot(np.linspace(1980,2050),np.linspace(1980,2050)*0+0.1, 'ko', alpha=0.2)

    #Title and labels
    ax.set_title('Prediction for Concentration of CO2 at Ohaaki with uncertainty')
    ax.set_ylabel('Concentration')
    ax.set_xlabel('Time (yr)')
    plt.savefig('Concentration_Predictions_Uncertainty.png')

if __name__ == "__main__":
    main()