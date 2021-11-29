import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
from math import ceil
import numpy as np


def solve_one_time_step(u_0, mu):
    def create_main_matrix(n_x_points, mu):
        """
        Matrix for theta method
        """
        tri_diag = np.ones((3, n_x_points))
        tri_diag[1] = -2 * tri_diag[1]
        a_matrix = sparse.spdiags(tri_diag, [-1, 0, 1], n_x_points, n_x_points) * mu

        i_matrix = sparse.identity(n_x_points)
        return a_matrix, i_matrix

    u = u_0

    D2, I = create_main_matrix(n_x_points=u_0.shape[0], mu=mu)
    lhs = (I - D2 / 2)
    rhs = (I + D2 / 2) * u
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
    dx = 1 / n_x_points
    dt = dx ** 2 * mu / c
    n_t_points = ceil(t_final / dt)

    x = np.linspace(x_a, x_b, n_x_points)
    t = np.arange(0, t_final, dt)
    u_0 = np.reshape(u_0_func(x), (100, 1))
    data = [u_0]

    u = u_0
    for t_i in range(n_t_points):
        u = solve_one_time_step(u_0=u, mu=mu)
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
    return -6 * np.sin(np.pi * x)


def moving_boundary(t):
    """ to be defined """
    pass


solve_heat_equation(u_0_func=initial_value,
                    t_final=0.1,
                    x_a=0,
                    x_b=1,
                    temp_a=0,
                    temp_b=0,
                    n_x_points=100,
                    c=1,
                    plot=True)
