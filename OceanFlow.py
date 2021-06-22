from operator import itemgetter

import pandas as pd
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import KFold
from PIL import Image
import OceanFlow_utils

rng = default_rng(12345)
plt.style.use("seaborn-whitegrid")


def interpolate(arr, n, l2, sig2, tau, plot=False):
    test = np.linspace(0, 100, n)
    train = np.linspace(0, 100, len(arr))

    K_ss = OceanFlow_utils.compute_kernel(test, l2, sig2)
    L = np.linalg.cholesky(K_ss + tau * np.eye(n))
    f_prior = np.dot(L, np.random.normal(size=(n, 3)))

    K = OceanFlow_utils.compute_kernel(train, l2, sig2)
    L = np.linalg.cholesky(K + tau * np.eye(len(train)))

    K_s = OceanFlow_utils.compute_kernel(train, l2, sig2, test)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, arr)).reshape((n,))  # This is where labels come in (arr)

    s2 = np.diag(K_ss) - np.sum(Lk ** 2, axis=0)
    stdv = np.sqrt(s2)

    L_final = np.linalg.cholesky(K_ss + tau * np.eye(n) - np.dot(Lk.T, Lk))
    f_post = mu.reshape(-1, 1) + np.dot(L_final,
                                        np.random.normal(size=(n, 1)))  # mu + beta*N(0,I), draws random samples
    if plot:
        plt.plot(train, arr, c='crimson', lw=3, label="Ground Truth")
        plt.plot(test, f_post, c='darkgreen', label="Predicted F_Posterior")
        plt.gca().fill_between(test.flat, mu - 3 * stdv, mu + 3 * stdv, color="#dddddd")
        plt.plot(test, mu, 'r--', lw=3, label="Mu - Average", c="darkblue")
        title = "U Direction" if arr is u else "V Direction Interpolation"
        plt.title(title + f" tau = {tau}, l^2 = {l2}, sig^2 = {sig2}")
        plt.legend()
        plt.show()

    return f_post, mu, stdv


# f_post, mu, stdv = interpolate(u, 300, l2, sig2, tau)


# def compute_movement_extended(x_current, y_current, n):
#     # computes new movements and locations for each time step, given x and y initial coordinates
#     x, y, u, v = [np.zeros(n) for _ in range(4)]
#     for i in range(n):
#         x[i] = min(x_current, 554)
#         y[i] = min(y_current, 503)
#         int_x = int(round(x[i], 0))
#         int_y = int(round(y[i], 0))
#         f_post_u, mu_u, stdv_u = interpolate(u_array[int_x, int_y, :], n, l2, sig2, tau)
#         f_post_v, mu_v, stdv_v = interpolate(v_array[int_x, int_y, :], n, l2, sig2, tau)
#         u[i] = f_post_u[i]
#         v[i] = f_post_v[i]
#         x_current = x_current + u[i] * 24  # km = km + km/hr * 24 hr/day. Each time step is 1 day in f_post
#         y_current = y_current + v[i] * 24
#
#     # movement_summary = pd.DataFrame({'x': x, 'y': y, 'u': u, 'v': v}) # row is timestep, columns are x coord, y coord, x magnitude, y magnitude
#     movement_summary = np.stack((x, y, u, v), axis=1)
#     return movement_summary


def Kfold_function(direction_vec, l2=None, sig2=None, tau=None, hyperparameters=None, plot=False):
    likelihood = []
    if hyperparameters is not None:
        l2 = hyperparameters["l2"]
        sig2 = hyperparameters["sig2"]
        tau = hyperparameters["tau"]
    GKF = KFold(n_splits=20)
    for train, test in GKF.split(direction_vec):
        train_labels = direction_vec[train]
        train = train.reshape(-1, 1)
        test = test.reshape(-1, 1)
        train_len, test_len = len(train), len(test)
        K_train = OceanFlow_utils.compute_kernel(train, train, l2, sig2)
        K_test = OceanFlow_utils.compute_kernel(test, test, l2, sig2)
        K_cross = OceanFlow_utils.compute_kernel(train, test, l2, sig2)
        L = np.linalg.cholesky(K_train + tau * np.eye(train_len))
        Lk = np.linalg.solve(L, K_cross)
        mu = np.dot(Lk.T, np.linalg.solve(L, train_labels))
        stdv = np.sqrt(np.diag(K_test) - np.sum(Lk ** 2, axis=0))
        L_final = np.linalg.cholesky(K_test + tau * np.eye(test_len) - np.dot(Lk.T, Lk))
        f_post = mu.reshape(-1, 1) + L_final @ np.random.normal(size=(test_len, 1))
        log_like = OceanFlow_utils.get_log_likelihood(f_post, train_labels, K_train, K_test, K_cross, tau)
        likelihood.append(log_like)
        if plot:
            plt.plot(np.arange(len(direction_vec)), direction_vec, lw=3)
            plt.plot(test, f_post)
            plt.gca().fill_between(test.flat, mu - 3 * stdv, mu + 3 * stdv, color="#dddddd")
            plt.plot(test, mu, 'r--', lw=2)
            plt.show()

    return round(np.array(likelihood).mean(), 2)



def hyperparameter_optimization(direction_vector, direction, hyperparameters):
    tau = hyperparameters["tau"]
    l2 = hyperparameters["l2"]
    sig2 = hyperparameters["sig2"]

    Kfold_vectorized = np.vectorize(Kfold_function, excluded={0, "plot", "hyperparameters"})

    l2_mesh, sig2_mesh = np.meshgrid(l2, sig2)
    log_like = Kfold_vectorized(direction_vector, l2=l2_mesh, sig2=sig2_mesh, tau=tau)

    log_like_dict = {}
    for l_iter, s_iter, log_iter in zip(l2_mesh.flatten(), sig2_mesh.flatten(), log_like.flatten()):
        log_like_dict[(l_iter, s_iter)] = log_iter

    max_likelihood = {"max_likelihood": np.max(log_like)}
    max_log_index = np.where(log_like == max_likelihood["max_likelihood"])
    max_likelihood["l2"] = np.round(l2_mesh[max_log_index[0], max_log_index[1]],1)
    max_likelihood["sig2"] = np.round(sig2_mesh[max_log_index[0], max_log_index[1]],1)
    max_likelihood["tau"] = tau

    fig, ax = plt.subplots()
    sns.heatmap(log_like, ax=ax)
    ax.set_xticklabels(list(l2))
    ax.set_yticklabels(list(sig2))
    ax.set_xticks(np.arange(len(l2)))
    ax.set_yticks(np.arange(len(sig2)))
    ax.set_xlabel("l2")
    ax.set_ylabel("sig2")
    ax.set_title(f"{direction} Optimum: l2 - {max_likelihood['l2']} sig2 - {max_likelihood['sig2']}")
    plt.savefig(f"OceanFlowImages/{direction.upper()}_Parameter_Optimization_Heatmap.png", format="png")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(l2_mesh, sig2_mesh, log_like, cmap=cm.coolwarm)
    ax.set_title(f"{direction} Optimum: l2 - {max_likelihood['l2']} sig2 - {max_likelihood['sig2']}")
    ax.set_xlabel("l2")
    ax.set_ylabel("sig2")
    ax.set_zlabel("Log Likelihood")
    norm = mpl.colors.Normalize(vmin=np.min(log_like), vmax=max_likelihood["max_likelihood"])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm), ax=ax)
    plt.savefig(f"OceanFlowImages/{direction.upper()}_Parameter_Optimization.png", format="png")
    plt.show()
    return log_like_dict, max_likelihood


def kernel_heatmap(dim, l2, sig2):
    """Enter l2 and s2 as nxn tuples"""
    dummy_vector = np.arange(dim).reshape(-1, 1)
    fig, axes = plt.subplots(nrows=len(sig2), ncols=len(l2))
    for l2_outer, s2_outer, ax_outer in zip(l2, sig2, axes):
        for l2_inner, s2_inner, ax_inner in zip(l2_outer, s2_outer, ax_outer):
            kernel = OceanFlow_utils.compute_kernel(dummy_vector, dummy_vector, l2_inner, s2_inner)
            ax_inner.set_title(f"l2 = {l2_inner}, sig2 = {s2_inner}")
            sns.heatmap(kernel, ax=ax_inner, square=True, cmap="Blues", vmax=sig2.max())
    plt.suptitle(f"N = {dim} Kernel Covariance Matrix")
    fig.tight_layout()
    plt.savefig("OceanFlowImages/Kernel_Heatmap.png", format='png')
    plt.show()


def plot_crash_coordinates_gauss_prior(u_3d, v_3d, plane_crash_coordinates, hyperparameters):
    """Will only use the first two parameters of l2 and sig2"""
    l2 = hyperparameters["l2"][0:2]
    sig2 = hyperparameters["sig2"][0:2]
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    x, y = plane_crash_coordinates
    for array, dir, color, ax, param_l, param_s in zip([u_3d, v_3d], ["U", "V"], ['b', 'r'], axes, l2, sig2):
        velocity = array[x, y, :]
        t = len(velocity)
        time = np.arange(t).reshape(-1, 1)
        K_ss = OceanFlow_utils.compute_kernel(time, time, param_l, param_s)
        L = np.linalg.cholesky(K_ss + 1e-15 * np.eye(t))
        f_prior = np.dot(L, rng.normal(size=(t, 3)))
        ax.plot(time, velocity, c=color, linewidth=1.5)
        ax.plot(time, f_prior, linewidth=.75)
        ax.set_title(f"{dir} Direction; l2 = {param_l}, sig2 = {param_s}")
    plt.suptitle(f"Gaussian Prior for [{x}, {y}]")
    plt.savefig(f"OceanFlowImages/gaussian_prior.png", format="png")
    plt.show()


def plot_crash_coordinates_gauss_posterior(u_3d, v_3d, plane_crash_coordinates, hyperparameters):
    """Will only use the first two parameters of l2 and sig2"""
    l2 = hyperparameters["l2"][0:2]
    sig2 = hyperparameters["sig2"][0:2]
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    x, y = plane_crash_coordinates
    n = 1000
    for array, dir, color, ax, l_param, s_param in zip([u_3d, v_3d], ["U", "V"], ['b', 'r'], axes, l2, sig2):
        Ytrain = array[x, y, :]  # aka velocity
        t = len(Ytrain)
        Xtrain = np.arange(t).reshape(-1, 1)  # aka time
        Xtest = np.linspace(0, t, n).reshape(-1, 1)
        K = OceanFlow_utils.compute_kernel(Xtrain, Xtrain, l_param, s_param)
        K_ss = OceanFlow_utils.compute_kernel(Xtest, Xtest, l_param, s_param)
        K_s = OceanFlow_utils.compute_kernel(Xtrain, Xtest, l_param, s_param)
        L = np.linalg.cholesky(K + .00005 * np.eye(t))  # 100 x 100
        Lk = np.linalg.solve(L, K_s)  # 100 x 1000
        mu = np.dot(Lk.T, np.linalg.solve(L, Ytrain)).reshape((n,))  # 1000
        s2 = np.diag(K_ss) - np.sum(Lk ** 2, axis=0)
        stdv = np.sqrt(s2)
        L = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n) - np.dot(Lk.T, Lk))  # 1000 x 1000
        f_post = mu.reshape(-1, 1) + np.dot(L, np.random.normal(size=(n, 3)))  # 1000 x 3
        ax.plot(Xtrain, Ytrain, c=color, linewidth=1.5)
        ax.plot(Xtest, f_post, linewidth=.75)
        ax.set_title(f"{dir} Direction; l2 = {l_param}, sig2 = {s_param}")
        ax.fill_between(Xtest.flat, mu - 2 * stdv, mu + 2 * stdv, color="#dddddd")
    plt.suptitle(f"Gaussian Posterior for [{x}, {y}]")
    plt.savefig(f"OceanFlowImages/gaussian_post.png", format="png")
    plt.show()


def plane_crash(u_3d, v_3d, mask, plane_crash_coordinates, variance=10, num_points=25):
    """Given an x and y input at t = 0, simulates the flow of a theoretical 'plane crash'
    Hypothetically, if a plane were to crash into the ocean at point (x,y), where would you look for the part?
    This simulation will create an animation that shows you where to look at a given time step (PlaneSearch.gif)
    First, it creates num_points # of random coordinates that are distributed randomly according to a Gaussian
    distribution specified by mu_x, mu_y, and variance (sigma is assumed to be the same for x and y directions).
    It then computes the flow path of each of those points using the utility function compute_movement
    Plots the resulting data on top of the mask and animates it for 100 time steps
    """
    mu_x, mu_y = plane_crash_coordinates
    coordinates = OceanFlow_utils.get_coordinates_toy(num_points, mu_x, mu_y, variance)
    colors = OceanFlow_utils.get_colors(len(coordinates))

    _, _, time = np.shape(u_3d)

    tx, ty = [np.zeros([time, len(coordinates)]) for _ in range(2)]

    for i, (x, y) in enumerate(coordinates):
        movement = OceanFlow_utils.compute_movement(x, y, u_3d, v_3d, time)
        tx[:, i] = movement[:, 0]
        ty[:, i] = movement[:, 1]

    fig, ax = plt.subplots()
    ax.scatter([], [])
    ax.imshow(mask, alpha=0.5, cmap='ocean', aspect='auto')

    def animate2(i):
        ax.set_title(f"T = {i + 1}, mu_x = {mu_x}, mu_y = {mu_y}, var_xy = {variance}")
        ax.scatter(tx[i, :], ty[i, :], c=colors, s=2)
        return ax

    ani = animation.FuncAnimation(fig, animate2, frames=time, interval=50, repeat=False)
    ani.save("PlaneSearch.gif", writer=animation.PillowWriter(fps=10))
    plt.show()


def find_dipoles(u_3d, v_3d, mask, plot=False):
    """Returns a dictionary of dipoles (keys) and  correlation coefficients (values)

    Runs the following algorithm 100,000 times
    1. Create 4 random integers within the bounds of the grid
        a. (x1, y1; x2, y2), which correspond to two pairs of points on the ‘u’ grid
    2. Find the arrays corresponding to the velocity at those points for t = {1,…,100}
        a. Yields two 1x100 arrays
        b. if either are zero vectors, end this iteration
    3. Compute the correlation for these ‘u’ grid vectors
    4. Cutoff criteria
        i. If the absolute value of the correlation coefficient is > .9
        ii. x distances are greater than 10 grid units (30 km)
        iii. y distances are greater than 10 grid units (30 km)
    5. If those are met, compute the correlation coefficient at the same spot for the ‘v’ vector
        i. If the absolute value of the correlation coefficient is > .9, then add to dictionary
    6. Key = (x1, y1, x2, y2), value = (u_corr_coeff, v_corr_coeff)
        i. Can be positively or negatively correlated
    7. If plot, plots the dipoles on the map with a + or - sign if they are positively or negatively correlated
    """

    def compute_correlations(point_vec_1, point_vec_2):
        """arr is a 1D array (1x100) for point x,y for times T = 1 - 100"""
        if np.all(point_vec_1) == 0.0 or np.all(point_vec_2) == 0.0:
            return 0
        return (np.corrcoef(point_vec_1, point_vec_2)[0, 1]).round(2)

    samples = range(100000)
    correlations_dict = {}
    rows, cols, time = np.shape(u_3d)
    for _ in samples:
        col_index_1 = rng.integers(low=0, high=cols)  # This is the x direction (left-right). 555 columns
        col_index_2 = rng.integers(low=0, high=cols)
        row_index_1 = rng.integers(low=0, high=rows)  # This is the y direction (top-bottom). 504 rows
        row_index_2 = rng.integers(low=0, high=rows)
        point_vec_a = u_3d[row_index_1, col_index_1, :]
        point_vec_b = u_3d[row_index_2, col_index_2, :]
        # The reason this is y then x is the array is indexed by rows, columns
        # and the y direction specifies the # of rows and x direction the # of columns

        if point_vec_a.all() == 0.0 or point_vec_b.all() == 0.0:
            continue

        corr1 = compute_correlations(point_vec_a, point_vec_b)

        cutoff = [abs(corr1) > .90,
                  abs(col_index_1 - col_index_2) > 10,
                  abs(row_index_1 - row_index_2) > 10]

        if np.all(cutoff):
            point_vec_c = v_3d[row_index_1, col_index_1, :]
            point_vec_d = v_3d[row_index_2, col_index_2, :]
            corr2 = compute_correlations(point_vec_c, point_vec_d)

            if abs(corr2) > .90 and np.sign(corr1) == np.sign(corr2):
                key = (col_index_1, row_index_1, col_index_2, row_index_2)
                correlations_dict[key] = (corr1, corr2)
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(mask, alpha=0.5, cmap='ocean', aspect='auto')
        for key, value in correlations_dict.items():
            x1, y1, x2, y2 = key
            corr_u, corr_v = value
            print(f"Point 1: ({x1},{y1}); Point 2: ({x2},{y2}); "
                  f"u-corr coefficient: {corr_u}; v-corr coefficient: {corr_v}")
            marker = "+" if corr_u > 0 else "_"
            ax.scatter([x1, x2], [y1, y2], marker=marker, s=80)
        plt.savefig("Dipole.png", format='png', transparent=True)
        plt.show()

    return correlations_dict


def ocean_streamplots(u, v, mask):
    """Plots & animates the velocity-weighted flow using Matplotlib Streamplots with a mask background
    There is still a bug in the code that makes the gif resize at certain time points, but it's time to move on for now
    """
    x, y, time = np.shape(u)

    fig, ax = plt.subplots()

    def stream_animate(t):
        """Returns streamplot at time = t to be used by FuncAnimation"""
        plt.cla()
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(mask, alpha=0.5, cmap='ocean', aspect='auto')
        ax.set_aspect('equal')
        u_m = np.ma.array(u[:, :, t], mask=~mask)
        v_m = np.ma.array(v[:, :, t], mask=~mask)
        velocity = OceanFlow_utils.calc_velocity(u_m, v_m)
        ax.set_title(f"Velocity-Weighted Streamplot at t={t}")
        lw = 4 * velocity / velocity.max()
        ax.streamplot(x, y, u_m, v_m, density=2, linewidth=lw)
        fig.tight_layout()
        return ax

    ani = animation.FuncAnimation(fig, stream_animate, frames=time, interval=200, repeat=False)
    ani.save("Streamplots.gif", writer=animation.PillowWriter(fps=4))
    plt.show()


def main():
    """ [Insert description]
        1. Tries to load numpy arrays from saved files. If not found, calls load_save_data from OceanFlow_utils
        2. Calls the ocean_streamplot function to create and save a streamplot of the flow data"""
    # TODO Docstring everything with triple quotes then enter
    try:
        uv_mask_data = np.load("u_3d.npy"), np.load("v_3d.npy"), np.load("mask.npy").astype('bool')
    except FileNotFoundError:
        OceanFlow_utils.load_save_data()
        uv_mask_data = np.load("u_3d.npy"), np.load("v_3d.npy"), np.load("mask.npy").astype('bool')

    u_3d, v_3d, mask = uv_mask_data
    rows, columns, time = np.shape(u_3d)

    # -------------------------------------------------------------------------------------------------------------#
    # TODO May not need this because imshow can plot with mask
    filename = "mask.png"
    try:
        mask_image = Image.open(filename)
    except FileNotFoundError:
        OceanFlow_utils.create_mask_image(mask, filename)
        mask_image = Image.open(filename)
    # -------------------------------------------------------------------------------------------------------------#

    # TODO Commenting this out for now because it takes a long time to run. Uncomment for final version ###
    # ocean_streamplots(*uv_mask_data)

    # -------------------------------------------------------------------------------------------------------------#
    # TODO Commenting this out for now because it takes a long time to run. Uncomment for final version ###
    # correlations = find_dipoles(*uv_mask_data, plot=True)

    # -------------------------------------------------------------------------------------------------------------#

    # Say you have this data and a plane crashed at 400, 400. Where do you look for parts? Let's create as simulation
    # that shows where the flow possibly took the parts as a function of the timestep and assumed variance
    # TODO Commenting this out for now because it takes a long time to run. Uncomment for final version ###
    plane_crash_coordinates = [400, 400]
    # plane_crash(*uv_mask_data, plane_crash_coordinates)

    # -------------------------------------------------------------------------------------------------------------#
    # The time steps are quite far apart, so you want to interpolate between seemingly random signals. How do you do
    # this? Regression methods are parametric and random noise is infinitely parametric. Where would you even begin?
    # Enter Gaussian Process, a non-parametric approach that "defines a prior over functions, which can be converted
    # into a posterior over functions once we have seen some data"
    #
    # Ocean flow is a great candidate for being modeled by a Gaussian process because the noisy signal requires a
    # more unconstrained approach to modeling than any regression could provide. Ocean flow is a function of numerous
    # random elements like temperature, barometric pressure, cloud cover, etc. The sum off all these --> Gaussian

    # Our "prior" estimate before any data is observed is just a Gaussian distribution with mean = 0 and an arbitrary
    # covariance between points. This covariance matrix can be approximated by a what's called a Kernel, which gives
    # us a measure of "similarity" between all points in a set. There are certain hyperparameters,
    # namely the length scale l^2 and sig^2 (get name) that define the Kernel.
    # For this example, we will use the radial basis function (RBF) kernel because it is infinitely differentiable
    # (scaled to infinite dimension) and fits noisy models really well.
    # The key assumption here is that points close in time indices should be correlated with each other. Therefore,
    # if we know the flow at say times t = 4 and t = 5, we figure the flow at t = 4.5 will be somewhere in between.
    # And t=4.5 also has some decaying dependence on t = 3, t = 2, etc. We need to map this covariance somehow
    # Th next step will be to optimize the model by finding the best hyperparamters.
    # Note, that in this example, all the data points are equally spaced, but that need not be true.

    # First, let's get a sense of what the prior distribution looks like with arbitrary l2 and s2. This is the model
    # before any data has been incorporated
    l2, sig2 = (10, 1), (1, .1)
    hyperparameters_gaussian_plot = {"l2": l2, "sig2": sig2}
    # plot_crash_coordinates_gauss_prior(u_3d, v_3d, plane_crash_coordinates, hyperparameters_gaussian_plot)

    # It's clear that a higher length scale smooth smooths out the curves and a lower sigma makes the distribution
    # tighter. This model assumed the mean = 0, which is clearly not the case. We can incorporate our measurement data
    # and find a posterior Gaussian model. While the first plot looks like it fits the data well, it is actually a poor
    # model, as it overfits the data.

    # Let's take a look at what the posterior function looks like with the same hyperparameters. In other words,
    # after we see some data, how has the model improved?
    plot_crash_coordinates_gauss_posterior(u_3d, v_3d, plane_crash_coordinates, hyperparameters_gaussian_plot)

    # -------------------------------------------------------------------------------------------------------------#

    # Isolate the data to only look at a 2D array at the specified crash point instead of the entire set
    crash_u = u_3d[plane_crash_coordinates[0], plane_crash_coordinates[1], :]
    crash_v = v_3d[plane_crash_coordinates[0], plane_crash_coordinates[1], :]

    # Let's dive deeper into the Kernel function and what it is telling us
    # The kernel function will input an Nx1 matrix and return a symmetric NxN matrix with all diagonals = 1
    # and each i,j entry is the covariance between points i,j

    # Let's visualize this
    # heatmap_l2 = np.array(((10, 5, 1), (10, 5, 1), (10, 5, 1)))
    # heatmap_sig2 = np.array(((10, 10, 10), (8, 8, 8), (5, 5, 5)))
    # kernel_heatmap(10, heatmap_l2, heatmap_sig2)

    # We can see that the length scale defines how quickly the covariance decays as the distance between points gets
    # further away. We can also see that sigma^2 defines the magnitude of the covariance. The diagonals will be
    # equal to sigma^2 and decay from there. A higher covariance means a wider, more noisier looking trend.

    # -------------------------------------------------------------------------------------------------------------#

    # So what are the hyperparameters that will yield the best model? For that, we need to define a metric by which
    # to gauge our predictions - a measure of how far off the predictions are from ground truth. We will use the
    # log-likelihood as the measurement of how good the fit is by using K-Fold cross-validation. The idea here is to
    # artificially remove some of the data from our ground truth training data, make the prediction with a given set
    # of hyperparameters, and compare the goodness of fit by using log-likelihood estimation - or what is the
    # probability of seeing our prediction given the training data, training covariance, and cross-covariance. We
    # repeat this process for all sets of hyperparameters and for 25 K-folds of the data.
    # [Insert explanation about log likelihood]

    # K_fold_hyperparameters = {"l2": 10, "sig2":.05, "tau":1e-5}
    # loss_list, avg_loss = Kfold_function(crash_u, K_fold_hyperparameters, plot=False)

    # l2_opt = np.arange(50, 200, 10)
    # sig2_opt = np.arange(30, 180, 10)
    # tau_opt = 1e-5
    # opt_hyperparameters = {"l2": l2_opt, "sig2": sig2_opt, "tau": tau_opt}
    # log_like_dict_u, max_likelihood_u_params = hyperparameter_optimization(crash_u, "U", opt_hyperparameters)
    # log_like_dict_v, max_likelihood_v_params = hyperparameter_optimization(crash_v, "V", opt_hyperparameters)
    #
    # print("U Direction Optimal Parameters")
    # for key, value in max_likelihood_u_params.items():
    #     print(f"{key}: {value}")
    #
    # print("V Direction Optimal Parameters")
    # for key, value in max_likelihood_v_params.items():
    #     print(f"{key}: {value}")
    #
    # # Our optimization algorithm has returned an optimal (l2, sig2) of (190, 60) for the U Direction and (190, 50).
    # # Marginal improvements could be made by increasing the granularity of our grid search, but at this point, I feel
    # # like these parameters are good enough for us to begin our model fit.
    # # -------------------------------------------------------------------------------------------------------------#
    # # Let's plot the Gaussian posteriors with these new parameters
    # max_likelihood_params = OceanFlow_utils.zip_dict(max_likelihood_u_params, max_likelihood_v_params)
    # plot_crash_coordinates_gauss_posterior(u_3d, v_3d, plane_crash_coordinates, max_likelihood_params)

    # -------------------------------------------------------------------------------------------------------------#




if __name__ == "__main__":
    main()
