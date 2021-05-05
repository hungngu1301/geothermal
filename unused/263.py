
from matplotlib import pyplot as plt    # MATPLOTLIB is THE plotting module for Python
from sdlab_functions import*
from numpy.linalg import norm, solve
tol = 1.e-6

# Set common timeline
plot_timeline = np.arange(1965,2018,0.1)

# Find eqn for PW1;
tw1,yw1 = np.genfromtxt('cs_c.txt',delimiter=',',skip_header=1).T 
A = spline_coefficient_matrix(tw1) 
b = spline_rhs(tw1,yw1)
a_coeffs = solve(A,b)

# Interpolate over common timeline.
yjw1 = spline_interpolate(plot_timeline,tw1,a_coeffs)

# Find eqn for PW1;
tw2,yw2 = np.genfromtxt('cs_p.txt',delimiter=',',skip_header=1).T 
A = spline_coefficient_matrix(tw2) 
b = spline_rhs(tw2,yw2)
a_coeffs2 = solve(A,b)

# Interpolate over common timeline.
yjw2 = spline_interpolate(plot_timeline,tw2,a_coeffs2)

# Find eqn for PW1;
tw3,yw3 = np.genfromtxt('cs_q.txt',delimiter=',',skip_header=1).T 
A = spline_coefficient_matrix(tw3) 
b = spline_rhs(tw3,yw3)
a_coeffs3 = solve(A,b)

# Interpolate over common timeline.
yjw3 = spline_interpolate(plot_timeline,tw3,a_coeffs3)


# Find eqn for PW1;
tw4,yw4 = np.genfromtxt('cs_cc.txt',delimiter=',',skip_header=1).T 
A = spline_coefficient_matrix(tw4) 
b = spline_rhs(tw4,yw4)
a_coeffs4 = solve(A,b)

# Interpolate over common timeline.
yjw4 = spline_interpolate(plot_timeline,tw4,a_coeffs4)


f,ax1 = plt.subplots(nrows=1,ncols=1)
ax2 = ax1.twinx()
ax1.set_title('Comparision of CO2 injection rate & CO2 weight fraction in Reservoir',y=1.08)
ax1.set_ylabel('CO2 Injection Rates (kg/s)') # Labelling the y axis 
ax2.set_ylabel('CO2 weight fraction in Reservior')
ax1.set_xlabel('Year')
ax1.plot(plot_timeline, yjw1, 'r--', label='CO2 Injection Rates [kg/s]')
ax2.plot(plot_timeline, yjw4, 'k-', label='CO2 weight fraction in Reservior')
#ax1.plot(plot_timeline, yjw3, 'k--', label='Fluid Extraction Rate (kg/s)')
#ax2.plot(plot_timeline, yjw4 , 'r-', label='Net Mass')
ax1.legend(bbox_to_anchor=(0., 1.02, 0.5, .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.) # Creating legend for plots on ax1
ax2.legend(loc=2)      
#plt.show(f)
plt.savefig('Mass Balance.png',dpi=300)