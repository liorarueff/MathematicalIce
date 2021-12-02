import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.sparse.linalg
from math import ceil
import numpy as np
import sys


def solve_one_time_step(u_0, mu_vec, temp_a=0, temp_b=0):
    print("h")
    def create_main_matrix(n_x_points, mu_vec):
        """
        Matrix for theta method
        """
        tri_diag = np.ones((3, n_x_points))
        tri_diag[1] = -2 * tri_diag[1]

        for row in range(n_x_points):
            tri_diag[:, row] *= float(mu_vec[row])

        a_matrix = sparse.spdiags(tri_diag, [-1, 0, 1], n_x_points, n_x_points)

        i_matrix = sparse.identity(n_x_points)
        return a_matrix, i_matrix

    u = u_0

    bv = np.zeros_like(u_0)
    bv[0] = mu_vec[0] * temp_a
    bv[-1] = mu_vec[0] * temp_b

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
    dx = (x_b - x_a) / n_x_points
    dt = dx ** 2 * mu / c
    n_t_points = ceil(t_final / dt)

    x = np.linspace(x_a, x_b, n_x_points)
    t = np.arange(0, t_final, dt)
    u_0 = np.reshape(u_0_func(x), (100, 1))
    data = [u_0]

    u = u_0
    for t_i in range(n_t_points):
        u = solve_one_time_step(u_0=u, mu=mu, temp_a=temp_a - 1 + np.cos(t_i * dt),
                                temp_b=temp_b - 1 + np.cos(t_i * dt))
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
    return -6 * np.sin(np.pi * (x - 0.5)) + 2 * (x - 0.5)


def find_zeros(y_arr, a, b):
    """
    Returns the x-values (assuming y_arr is on a linear interpolation mesh between a and b) where the y_arr mesh
    function changes sign.
    """
    zeros_i = []

    for i in range(len(y_arr) - 1):
        if y_arr[i] * y_arr[i + 1] < 0:  # This means that there is a sign change.
            zeros_i.append(i)  # We want to store the index

    # Let's now translate these indices into x values.
    dx = (b - a) / len(y_arr)
    zeros = []
    for index in zeros_i:
        zeros.append((index + 0.5) * dx)  # Adding half a step because the zero is between i and i+1.

    return zeros


# def find_zeros_func(f: callable, a, b):
#     k = 1
#     xs = np.linspace(a, b, 1000*k)
#     t_zero = f(xs)
#     sgn = np.sign(t_zero)
#     zbd = []
#
#     for i in range(0,len(sgn)-1):
#         if sgn[i] != sgn[i+1]:
#             zbd.append((xs[i]+xs[i+1])/2)
#
#     while len(zbd) != 2 and k < 11:
#         k += 1
#         xs = np.linspace(a, b, 1000 * k)
#         t_zero = f(xs)
#         sgn = np.sign(t_zero)
#         zbd = []
#         for i in range(0, len(sgn) - 1):
#             if sgn[i] != sgn[i + 1]:
#                 zbd.append((xs[i] + xs[i + 1]) / 2)
#
#     if len(zbd) != 2:
#         sys.exit("The function u_0 might not be a suitable choice. The function u_0 must be continuous and have exactly two zeros in [x_a,x_b]")
#     h1 = zbd[0]
#     h2 = zbd[1]
#     h = [h1, h2]
#
#     return h


# def find_zeros_array(u, a, b, tol):
#     k = len(u)
#     xs = np.linspace(a, b, k)
#     sgn = np.sign(u)
#     zbd = []
#     zbd_id = []
#     h = []
#
#     for i in range(0,len(sgn)-1):
#         if sgn[i] != sgn[i+1]:
#             zbd.append(xs[i])
#             zbd_id.append(i)
#
#     if len(zbd) == 1:
#         if abs(u[zbd_id[0]]) < tol:
#             h.append(xs[zbd_id[0]])
#             h.append(xs[zbd_id[0]])
#         else:
#             h.append((xs[zbd_id[0]] + xs[zbd_id[0] + 1]) / 2)
#             h.append((xs[zbd_id[0]] + xs[zbd_id[0] + 1]) / 2)
#     elif len(zbd) == 2:
#         if abs(u[zbd_id[0]]) < tol:
#             h.append(xs[zbd_id[0]])
#         else:
#             h.append((xs[zbd_id[0]]+xs[zbd_id[0]+1])/2)
#         if abs(u[zbd_id[1]]) < tol:
#             h.append(xs[zbd_id[1]])
#         else:
#             h.append((xs[zbd_id[0]]+xs[zbd_id[0]+1])/2)
#     else:
#         h = []
#
#     return h


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
    dx = (x_b - x_a) / n_x_points
    dt = t_final / n_t_points
    mu1 = c1 * dt / dx ** 2
    mu2 = c2 * dt / dx ** 2
    mu3 = c3 * dt / dx ** 2

    x = np.linspace(x_a, x_b, n_x_points)
    t = np.arange(0, t_final, dt)
    u_0 = np.reshape(u_0_func(x), (100, 1))
    data = [u_0]

    # bd1 = []
    # bd2 = []
    # u_0 = u_0_func()
    # h = find_zeros(u_0_func, x_a, x_b)
    # bd1.append(h[0])
    # bd2.append(h[1])
    h_1_arr = []
    h_2_arr = []

    h_data = find_zeros(u_0, a=x_a, b=x_b)
    print("Starting boundary points: ", h_data)
    h_1_arr.append(h_data[0])
    h_2_arr.append(h_data[1])

    u = u_0
    for t_i in range(n_t_points):
        mu_vector = np.ones_like(u)

        mu_vector[[x < h_1_arr[-1]]] *= mu1
        mu_vector[np.logical_and(h_1_arr[-1] <= x, x < h_2_arr[-1])] *= mu2
        mu_vector[h_2_arr[-1] <= x] *= mu3

        u = solve_one_time_step(u_0=u, mu_vec=mu_vector, temp_a=temp_a, temp_b=temp_b)

        h_data = find_zeros(u, a=x_a, b=x_b)
        if len(h_data) == 0:
            h_1_arr.append(h_data[0])
            h_2_arr.append(h_data[1])

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
            temp_a=5,
            temp_b=9,
            n_x_points=100,
            c1=0.01,
            c2=0.04,
            c3=0.01,
            tol=10 ** (-10),
            n_t_points=500,
            plot=True)

# solve_heat_equation(u_0_func=initial_value,
#                     t_final=50,
#                     x_a=-1,
#                     x_b=2,
#                     temp_a=-2,
#                     temp_b=4,
#                     n_x_points=100,
#                     c=0.01,
#                     plot=True)
