import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.sparse.linalg
from math import ceil
import numpy as np
import sys


def solve_one_time_step(u_0, mu_vec, temp_a=0, temp_b=0):
    def create_main_matrix(n_x_points, mu_vec):
        """
        Matrix for theta method
        """
        tri_diag = np.ones((3, n_x_points))
        tri_diag[1] = -2 * tri_diag[1]
        for row in range(n_x_points):
            tri_diag[:, row] *= mu_vec[row]
        a_matrix = sparse.spdiags(tri_diag, [-1, 0, 1], n_x_points, n_x_points)

        i_matrix = sparse.identity(n_x_points)
        return a_matrix, i_matrix

    u = u_0

    bv = np.zeros_like(u_0)
    bv[0] = mu_vec[0]*temp_a
    bv[-1] = mu_vec[0]*temp_b

    D2, I = create_main_matrix(n_x_points=u_0.shape[0], mu_vec=mu_vec)
    lhs = (I - D2 / 2)
    rhs = (I + D2 / 2) * u + bv
    u = np.transpose(np.mat(sparse.linalg.spsolve(lhs, rhs)))

    return u


def solve_heat_equation(u_0_func, t_final, x_a, x_b, temp_a, temp_b, n_x_points, c, plot=False):
    """
    This function approximates a solution to the generic heat equation

    u_0_func: function of x that returns the initial value.
    t_final: Latest time to simulate to [s]
    x_a: The lowest x-value of the domain [m]
    x_b: The highest x-value of the domain [m]
    temp_a: The temperature at x=a (Dirichlet BV) [deg C]
    temp_b: The temperature at x=b (Dirichlet BV) [deg C]
    n_x_points: The number of points required in the x-direction.
    c: The constant in the heat equation.
    """
    mu = 1  # Arbitrarily chosen, pick a higher number to increase the time step.
    # This mu was initially set to 1/4 as it needed to be less than 1/2 for an explicit scheme.
    dx = (x_b-x_a)/n_x_points
    dt = dx ** 2 * mu / c
    n_t_points = ceil(t_final / dt)

    x = np.linspace(x_a, x_b, n_x_points)
    t = np.arange(0, t_final, dt)
    u_0 = np.reshape(u_0_func(x), (100, 1))
    data = [u_0]

    u = u_0
    for t_i in range(n_t_points):
        u = solve_one_time_step(u_0=u, mu=mu, temp_a=temp_a-1+np.cos(t_i*dt), temp_b=temp_b-1+np.cos(t_i*dt))
        data.append(u)

        if (t_i % 1000) == 0:
            print(".", end="")

    result = np.hstack(data)

    if plot:
        X, Y = np.meshgrid(x, t)
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Creating plot
        ax.plot_surface(X, Y, result[:, :-1].T)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("T [s]")
        plt.show()

    return result


def initial_value(x):
    return -6 * np.sin(np.pi * (x-0.5)) + 2*(x-0.5)


def find_zeros_func(f: callable, a, b):
    k = 1
    xs = np.linspace(a, b, 1000*k)
    t_zero = f(xs)
    sgn = np.sign(t_zero)
    zbd = []

    for i in range(0,len(sgn)-1):
        if sgn[i] != sgn[i+1]:
            zbd.append((xs[i]+xs[i+1])/2)

    while len(zbd) != 2 and k < 11:
        k += 1
        xs = np.linspace(a, b, 1000 * k)
        t_zero = f(xs)
        sgn = np.sign(t_zero)
        zbd = []
        for i in range(0, len(sgn) - 1):
            if sgn[i] != sgn[i + 1]:
                zbd.append((xs[i] + xs[i + 1]) / 2)

    if len(zbd) != 2:
        sys.exit("The function u_0 might not be a suitable choice. The function u_0 must be continuous and have exactly two zeros in [x_a,x_b]")
    h1 = zbd[0]
    h2 = zbd[1]
    h = [h1, h2]

    return h


def find_zeros_array(u, a, b, tol):
    k = len(u)
    xs = np.linspace(a, b, k)
    sgn = np.sign(u)
    zbd = []
    zbd_id = []
    h = []

    for i in range(0,len(sgn)-1):
        if sgn[i] != sgn[i+1]:
            zbd.append(xs[i])
            zbd_id.append(i)

    if len(zbd) == 1:
        if abs(u[zbd_id[0]]) < tol:
            h.append(xs[zbd_id[0]])
            h.append(xs[zbd_id[0]])
        else:
            h.append((xs[zbd_id[0]] + xs[zbd_id[0] + 1]) / 2)
            h.append((xs[zbd_id[0]] + xs[zbd_id[0] + 1]) / 2)
    elif len(zbd) == 2:
        if abs(u[zbd_id[0]]) < tol:
            h.append(xs[zbd_id[0]])
        else:
            h.append((xs[zbd_id[0]]+xs[zbd_id[0]+1])/2)
        if abs(u[zbd_id[1]]) < tol:
            h.append(xs[zbd_id[1]])
        else:
            h.append((xs[zbd_id[0]]+xs[zbd_id[0]+1])/2)
    else:
        h = []

    return h


def solve_model(u_0_func, t_final, x_a, x_b, temp_a, temp_b, n_x_points, c1, c2, c3, tol, n_t_points, plot=False):
    """
    u_0_func: function of x that returns the initial value.
    t_final: Latest time to simulate to [s]
    x_a: The lowest x-value of the domain [m], x_a = 0
    x_b: The highest x-value of the domain [m]
    temp_a: The temperature at x=a (Dirichlet BV) [deg C]
    temp_b: The temperature at x=b (Dirichlet BV) [deg C]
    n_x_points: The number of points required in the x-direction.
    c1: The constant in the heat equation in the first part.
    tol: Tolerance for zero finding.
    """
    # This mu was initially set to 1/4 as it needed to be less than 1/2 for an explicit scheme.
    dx = (x_b-x_a) / n_x_points
    dt = t_final / n_t_points
    mu1 = c1 * dt / dx ** 2
    mu2 = c2 * dt / dx ** 2
    mu3 = c3 * dt / dx ** 2


    bd1 = []
    bd2 = []
    h = find_zeros_func(u_0_func, x_a, x_b)
    bd1.append(h[0])
    bd2.append(h[1])

    x = np.linspace(x_a, x_b, n_x_points)
    t = np.arange(0, t_final, dt)
    u_0 = np.reshape(u_0_func(x), (100, 1))
    data = [u_0]

    u = u_0
    for t_i in range(n_t_points):
        def condition1(x):
            return x < bd1[-1]
        muidx1 = [idx for idx, element in enumerate(u) if condition1(element)]
        def condition2(x):
            return x < bd2[-1]
        muidx2 = [idx for idx, element in enumerate(u) if (not condition1(element) and condition2(element))]

        mu1s = np.ones((1, len(muidx1))) * mu1
        mu2s = np.ones((1, len(muidx2))) * mu2
        mu3s = np.ones((1, len(u)-len(muidx1)-len(muidx2))) * mu3


        mu_vec = np.concatenate((mu1s, mu2s, mu3s), axis=1)
        mu_vec = mu_vec[0]

        u = solve_one_time_step(u_0=u, mu_vec=mu_vec)

        # Find next point of functions h1 and h2
        h = find_zeros_array(u, x_a, x_b, tol)
        if len(h) == 2:
            bd1.append(h[0])
            bd2.append(h[1])
        else:
            bd1.append(bd1[-1])
            bd2.append(bd2[-1])

        data.append(u)

        if (t_i % 1000) == 0:
            print(".", end="")

    result = np.hstack(data)

    if plot:
        X, Y = np.meshgrid(x, t)
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Creating plot
        ax.plot_surface(X, Y, result[:, :-1].T)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("T [s]")
        plt.show()


solve_model(u_0_func=initial_value,
                    t_final=50,
                    x_a=0,
                    x_b=2,
                    temp_a=-2,
                    temp_b=4,
                    n_x_points=100,
                    c1=0.01,
                    c2=0.04,
                    c3=0.01,
                    tol=10**(-10),
                    n_t_points=500,
                    plot=True)

#solve_heat_equation(u_0_func=initial_value,
                    #t_final=50,
                    #x_a=-1,
                    #x_b=2,
                    #temp_a=-2,
                    #temp_b=4,
                    #n_x_points=100,
                    #c=0.01,
                    #plot=True)
