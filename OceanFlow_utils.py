import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng

"""Utility functions for loading, saving, and analyzing Ocean Flow data"""

rng = default_rng(12345)

def load_save_data():
    """ Creates a 3d array (x, y, t):(504, 555, 100) for both u and v directions; saves files as u_3d.npy and v_3d.npy
    Creates a 2D mask numpy array or grid (x, y):(504, 555) that indicates land vs ocean; saves file as mask.npy
    Each grid unit is equivalent to 3 kms
    Note - the OceanFlowData/*.csv files and *.npy files are in .gitignore to save space and improve performance
    """
    time = range(2, 101)
    u_3d = np.loadtxt("OceanFlowData/1u.csv", delimiter=',')
    v_3d = np.loadtxt("OceanFlowData/1v.csv", delimiter=',')
    mask = np.loadtxt("OceanFlowData/mask.csv", delimiter=',')

    for i in time:
        u_3d_open = np.loadtxt("OceanFlowData/" + str(i) + "u.csv", delimiter=',')
        u_3d = np.dstack((u_3d, u_3d_open))
        v_3d_open = np.loadtxt("OceanFlowData/" + str(i) + "v.csv", delimiter=',')
        v_3d = np.dstack((v_3d, v_3d_open))

    np.save("u_3d", u_3d, allow_pickle=True)
    np.save("v_3d", v_3d, allow_pickle=True)
    np.save("mask", mask, allow_pickle=True)


def calc_velocity(u, v):
    """Creates an array of combined velocities given arrays of u and v direction velocity"""
    return np.sqrt(np.square(u) + np.square(v))


def compute_movement(x_current, y_current, u_3d, v_3d, time):
    """Computes new movements and locations for each time step, given x and y initial coordinates"""
    x, y, u, v = [np.zeros(time) for _ in range(4)]
    for i in range(time):
        x[i] = x_current
        y[i] = y_current
        u[i] = u_3d[int(round(x_current, 0)), int(round(y_current, 0)), i]
        v[i] = v_3d[int(round(x_current, 0)), int(round(y_current, 0)), i]
        x_current = x_current + u[i]  # Time step is 3 hours and each grid space is 3 hours = 1 grid space per time unit
        y_current = y_current + v[i]

    movement_summary = np.stack((x, y, u, v), axis=1)
    return movement_summary


def get_coordinates_toy(length, mu_x, mu_y, var_xy):
    """Generates random points from the normal distribution at mu_* and var_xy"""
    x = rng.normal(mu_x, var_xy, length - 1)
    x = np.append(x, mu_x)
    y = rng.normal(mu_y, var_xy, length - 1)
    y = np.append(y, mu_y)
    coordinates = list(zip(x, y))
    return coordinates


def get_colors(length):
    """Returns s a list (length=length) of randomly created tuples to be used as the color argument in matplotlib"""
    rbg = []
    for i in range(length):
        rbg.append(tuple(rng.random(3, )))
    return rbg

def get_log_likelihood(posterior, train_labels, K_train, K_test, K_cross, tau):
    """
    :param posterior:
    :param train_labels:
    :param K_train:
    :param K_test:
    :param K_cross:
    :param tau:
    :return:
    C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning, the MIT Press, 2006 for more info
    """
    intermediate_term = K_cross.T @ np.linalg.inv(K_train + tau * np.eye(len(K_train)))  # 5x95 @ 95x95 = 5 x 95
    mu = intermediate_term @ train_labels.reshape(-1,1)  # 5x95 @ 95x1 = 5x1
    cov = K_test - intermediate_term @ K_cross  # 5x5 - 5x95 @ 95x5 = 5x5
    log_like = -.5 * np.log(cov) - (posterior - mu)**2/(2*cov) - .5*np.log(2*np.pi)
    return log_like
    # log_like = -.5 * posterior.T @ np.linalg.solve(K, posterior)\
    #           - .5 * np.log(np.linalg.det(K))\
    #           - 10 / 2 * np.log(2 * np.pi)
    # return log_like


def compute_kernel(vector_1, vector_2, l2, sig2):
    """ This computes the kernel between the INDICES of train and test, not the labels.
     The idea is that closer indices (x-axis) leads to higher covariance
    :param vector_1: Vector for either training or test data
    :param vector_2: Vector for either training or test data
    :param l2: length scale. The shorter the length scale, the faster the covariance decays with Euclidian distance
    :param sig2: Scaling factor that impacts absolute variance. Will be the diagonal entries of the covariance matrix.
    :return: The RBF or squared distance kernel as a covariance (vector 1 = vector 2)
    or cross-covariance (vector 1 != vector 2)
    """
    squared_dist = np.sum(vector_1**2,1).reshape(-1,1) + np.sum(vector_2**2,1) - 2*np.dot(vector_1, vector_2.T)
    K = sig2 * np.exp(-1/l2 * squared_dist)
    return K

def zip_dict(u_d, v_d):
    """Creates a new dictionary with the same keys as the input and the values combined into a tuple"""
    zipped_dict = {}
    for key in ["l2", "sig2", "tau"]:
        zipped_dict[key] = u_d[key], v_d[key]
    return zipped_dict

