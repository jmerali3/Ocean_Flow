import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# Utility functions for loading, saving, and analyzing Ocean Flow data

def load_save_data():
    """ Creates a 3d array (x, y, t):(504, 555, 100) for both u and v directions; saves files as u_3d.npy and v_3d.npy
    Creates a 2D mask numpy array or grid (x, y):(504, 555) that indicates land vs ocean; saves file as mask.npy
    Each grid unit is equivalent to 3 kms
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


def create_mask_image(mask, filename):
    """Creates a seaborn heatmap and saves it. This is intended to be an 'implot' background for other plots
        Filename must be in png format
    """
    sns.heatmap(mask, cbar=False, cmap='Blues')
    #ocean cmap
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('tight')
    plt.axis('off')
    plt.show()
    try:
        plt.savefig(filename, format='png', transparent=True)
    except:
        print("Please enter filename in .png format")


def calc_velocity(u, v):
    return np.sqrt(np.square(u) + np.square(v))


def compute_movement(x_current, y_current):
    """Computes new movements and locations for each time step, given x and y initial coordinates"""
    x, y, u, v = [np.zeros(100) for _ in range(4)]
    for i in range(100):
        x[i] = x_current
        y[i] = y_current
        u[i] = u_array[int(round(x_current, 0)), int(round(y_current, 0)), i]
        v[i] = v_array[int(round(x_current, 0)), int(round(y_current, 0)), i]
        x_current = x_current + u[i]  # Time step is 3 hours and each grid space is 3 hours = 1 grid space per time unit
        y_current = y_current + v[i]

    movement_summary = np.stack((x, y, u, v), axis=1)
    return movement_summary