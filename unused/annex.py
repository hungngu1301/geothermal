import numpy as np
import matplotlib.pyplot as plt

J_scale = .1

def objectif_function(theta):
    [x, y] = theta
    x0 = -.2
    y0 = .35
    sigma_x = 1.2
    sigma_y = .8
    return  1-np.exp(-(x-x0)**2/sigma_x**2-(y-y0)**2/sigma_y**2)
    
def line_search(f, theta, J):
    gamma = 0.
    f_0 = f(theta)
    fgamma = lambda g: f(theta-g*J)
    j = (fgamma(.01)-f_0)/0.01
    N_max = 500
    N_it = 0
    while abs(j) > 1.e-5 and N_it<N_max:
        gamma += -j
        f_i = fgamma(gamma)
        j = (fgamma(gamma+0.01)-f_i)/0.01
        N_it += 1
    return gamma
    
    
    
def plot_J0(f, theta0, J0):

    N = 501
    x = np.linspace(-1, 1., N)
    y = np.linspace(-1., 1., N)
    xv, yv = np.meshgrid(x, y)
    Z = np.zeros(xv.shape)
    for i in range(len(y)):
        for j in range(len(x)):
            Z[i][j] = f([xv[i][j], yv[i][j]])

    plt.clf()
    ax1 = plt.axes()
    ax1.contourf(xv, yv, Z, 21, alpha = .8)
    ax1.scatter(theta0[0], theta0[1], color='k', s = 20.)
    ax1.arrow(theta0[0], theta0[1], -J_scale*J0[0], -J_scale*J0[1])
    ax1.set_xlim(-1.,1.)
    ax1.set_ylim(-1.,1.)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    plt.savefig('1plot_J.png', bbox_inches = 'tight')
    plt.show()
    
def plot_step(f, theta0, J, theta1):

    N = 501
    x = np.linspace(-1, 1., N)
    y = np.linspace(-1., 1., N)
    xv, yv = np.meshgrid(x, y)
    Z = np.zeros(xv.shape)
    for i in range(len(y)):
        for j in range(len(x)):
            Z[i][j] = f([xv[i][j], yv[i][j]])

    plt.clf()
    ax1 = plt.axes()
    ax1.contourf(xv, yv, Z, 21, alpha = .8)
    ax1.scatter([theta0[0], theta1[0]], [theta0[1], theta1[1]], color='k', s = 20.)
    ax1.plot([theta0[0], theta1[0]], [theta0[1], theta1[1]], color='k', linestyle = '--')
    ax1.arrow(theta0[0], theta0[1], -J_scale*J[0], -J_scale*J[1])
    ax1.set_xlim(-1.,1.)
    ax1.set_ylim(-1.,1.)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    plt.savefig('2plot_step.png', bbox_inches = 'tight')
    plt.show()
    
def plot_J1(f, theta0, J0, theta1, J1):
    
    N = 501
    x = np.linspace(-1, 1., N)
    y = np.linspace(-1., 1., N)
    xv, yv = np.meshgrid(x, y)
    Z = np.zeros(xv.shape)
    for i in range(len(y)):
        for j in range(len(x)):
            Z[i][j] = f([xv[i][j], yv[i][j]])

    plt.clf()
    ax1 = plt.axes()
    ax1.contourf(xv, yv, Z, 21, alpha = .8)
    ax1.scatter([theta0[0], theta1[0]], [theta0[1], theta1[1]], color='k', s = 20.)
    ax1.plot([theta0[0], theta1[0]], [theta0[1], theta1[1]], color='k', linestyle = '--')
    ax1.arrow(theta0[0], theta0[1], -J_scale*J0[0], -J_scale*J0[1])
    ax1.arrow(theta1[0], theta1[1], -J_scale*J1[0], -J_scale*J1[1])
    ax1.set_xlim(-1.,1.)
    ax1.set_ylim(-1.,1.)
    ax1.set_aspect('equal')
    plt.savefig('3plot_J1.png', bbox_inches = 'tight')
    plt.show()
    
def plot_Ji(f, list_theta, list_J):
    
    N = 501
    x = np.linspace(-1, 1., N)
    y = np.linspace(-1., 1., N)
    xv, yv = np.meshgrid(x, y)
    Z = np.zeros(xv.shape)
    for i in range(len(y)):
        for j in range(len(x)):
            Z[i][j] = f([xv[i][j], yv[i][j]])

    plt.clf()
    # fig = plt.figure(figsize = [10., 10.])
    # ax1 = fig.add_subplot(111)
    ax1 = plt.axes()
    ax1.contourf(xv, yv, Z, 21, alpha = .8)
    ax1.scatter([theta[0] for theta in list_theta[1:-1]], [theta[1] for theta in list_theta[1:-1]], color='k', linestyle = '--')
    ax1.scatter(list_theta[0][0], list_theta[0][1], color='b', linestyle = '--', label = 'Initial values')
    ax1.scatter(list_theta[-1][0], list_theta[-1][1], color='g', linestyle = '--', label = 'Final values')
    ax1.plot([theta[0] for theta in list_theta], [theta[1] for theta in list_theta], color='k', linestyle = '--')
    for i in range(len(list_theta)-1):
        ax1.arrow(list_theta[i][0], list_theta[i][1], -J_scale*list_J[i][0], -J_scale*list_J[i][1])
    ax1.set_xlim(-1.,1.)
    ax1.set_ylim(-1.,1.)
    ax1.set_aspect('equal')
    plt.savefig('4plot_J_i.png', bbox_inches = 'tight')
    plt.show()



    
    