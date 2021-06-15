import pandas as pd
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from sklearn.model_selection import KFold
from PIL import Image
import OceanFlow_utils

rng = default_rng(12345)

def compute_kernel(arr, l_sq, sig_sq, arr2=None):
    # This computes the kernel between the INDICES of train and test, not the labels.
    # The idea is that closer indices (x-axis) leads to higher covariance
    if arr2 is not None:
        l1 = len(arr)
        l2 = len(arr2)
    else:
        l1 = l2 = len(arr)
        arr2 = arr

    K = np.zeros([l1, l2])
    for i in range(l1):
        for j in range(l2):
            K[i, j] = sig_sq * np.exp((arr[i] - arr2[j]) ** 2 / -l_sq)
    return K


def interpolate(arr, n, l2, sig2, tau, plot=False):
    test = np.linspace(0, 100, n)
    train = np.linspace(0, 100, len(arr))

    K_ss = compute_kernel(test, l2, sig2)
    L = np.linalg.cholesky(K_ss + tau * np.eye(n))
    f_prior = np.dot(L, np.random.normal(size=(n, 3)))

    K = compute_kernel(train, l2, sig2)
    L = np.linalg.cholesky(K + tau * np.eye(len(train)))

    K_s = compute_kernel(train, l2, sig2, test)
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


def compute_movement_extended(x_current, y_current, n):
    # computes new movements and locations for each time step, given x and y initial coordinates
    x, y, u, v = [np.zeros(n) for _ in range(4)]
    for i in range(n):
        x[i] = min(x_current, 554)
        y[i] = min(y_current, 503)
        int_x = int(round(x[i], 0))
        int_y = int(round(y[i], 0))
        f_post_u, mu_u, stdv_u = interpolate(u_array[int_x, int_y, :], n, l2, sig2, tau)
        f_post_v, mu_v, stdv_v = interpolate(v_array[int_x, int_y, :], n, l2, sig2, tau)
        u[i] = f_post_u[i]
        v[i] = f_post_v[i]
        x_current = x_current + u[i] * 24  # km = km + km/hr * 24 hr/day. Each time step is 1 day in f_post
        y_current = y_current + v[i] * 24

    # movement_summary = pd.DataFrame({'x': x, 'y': y, 'u': u, 'v': v}) # row is timestep, columns are x coord, y coord, x magnitude, y magnitude
    movement_summary = np.stack((x, y, u, v), axis=1)
    return movement_summary


def Kfold_function(data, l_sq, sig_sq, tau, plot=False):
    loss = []
    GKF = KFold(n_splits=25)
    # fpost_list = []
    # stdv_list = []
    # mu_list = []
    # test_list = []
    for train, test in GKF.split(data):
        train_labels = data[train]
        test_labels = data[test]
        n = len(test)
        K_train = compute_kernel(train, l_sq, sig_sq)
        K_test = compute_kernel(test, l_sq, sig_sq)
        K_cross = compute_kernel(train, l_sq, sig_sq, arr2=test)
        L = np.linalg.cholesky(K_train + tau * np.eye(len(train)))
        Lk = np.linalg.solve(L, K_cross)
        mu = np.dot(Lk.T, np.linalg.solve(L, train_labels))
        stdv = np.sqrt(np.diag(K_test) - np.sum(Lk ** 2, axis=0))
        L_final = np.linalg.cholesky(K_test + tau * np.eye(n) - np.dot(Lk.T, Lk))
        f_post = mu.reshape(-1, 1) + L_final @ np.random.normal(size=(n, 1))
        log_lik = -.5 * f_post.T @ np.linalg.solve(K_test, f_post) - .5 * np.log(
            np.linalg.det(K_test)) - 10 / 2 * np.log(2 * np.pi)
        loss.append(log_lik)
        # fpost_list.append(f_post)
        # stdv_list.append(stdv)
        # mu_list.append(mu)
        # test_list.append(test)
        if plot:
            plt.plot(train, train_labels, lw=3)
            plt.plot(test, f_post)
            plt.gca().fill_between(test.flat, mu - 3 * stdv, mu + 3 * stdv, color="#dddddd")
            plt.plot(test, mu, 'r--', lw=2)
            plt.show()

    return loss, round(np.array(loss).mean(), 2)  # , fpost_list, stdv_list, mu_list, test_list


# sig_sq = .001
# l_sq = 15
# _, _, fpost, stdv, mu, test = Kfold_function_2(u, l_sq= l_sq, sig_sq=sig_sq, tau=.001, plot=False)
#
# fig, ax = plt.subplots()
# ax.set_xlim([0, 100])
# ax.set_ylim([-.5, 1])
# ax.scatter([], [])
#
# def animate_K(i):
#     f = fpost[i]
#     s = stdv[i]
#     m = mu[i]
#     t = test[i]
#     ax.set_title(f"Dir = U, Fold = {i}, sig^2 = {sig_sq}, l^2 = {l_sq}")
#     ax.plot(t, f)
#     fig.gca().fill_between(t.flat, m - 3 * s, m + 3 * s, color="#dddddd")
#     ax.plot(t, m, 'r--', lw=2)
#     ax.plot(np.arange(0,len(u)),u, c='b')
#     return ax
#
#
# ani = animation.FuncAnimation(fig, animate_K, frames=len(fpost), interval=250, repeat=False)
# ani.save("KFold-U.gif", writer=animation.PillowWriter(fps=10))
# plt.show()


def parameter_optimization(direction):
    tau = .0001
    l_sq = [1, 5, 10, 15, 50]
    sig_sq = [.01, .1, 1, 10, 100]
    params = np.zeros([len(l_sq), len(sig_sq)])

    if direction.lower() == 'u':
        direction_is_U = True
    elif direction.lower() == 'v':
        direction_is_U = False
    else:
        return None
    arr = u if direction_is_U else v
    for l_ind, l in enumerate(l_sq):
        for s_ind, s in enumerate(sig_sq):
            loss, loss_avg = Kfold_function(arr, l, s, tau)
            params[l_ind, s_ind] = loss_avg

    df = pd.DataFrame(params, columns=sig_sq, index=l_sq)
    sns.heatmap(df, vmax=0, vmin=-20, annot=True)
    plt.ylabel('l^2')
    plt.xlabel('sig^2')
    title = "U Direction" if direction_is_U else "V Direction"
    plt.title(title)
    plt.show()


def heatmap(K):
    mask = np.zeros_like(K)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots()
        ax = sns.heatmap(K, mask=mask, square=True, cmap="YlGnBu")
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
    size_dict = OceanFlow_utils.get_3d_size_extent(u)
    x = np.arange(size_dict["columns"])
    y = np.arange(size_dict["rows"])
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

    ani = animation.FuncAnimation(fig, stream_animate, frames=size_dict["time"], interval=200, repeat=False)
    ani.save("Streamplots.gif", writer=animation.PillowWriter(fps=4))
    plt.show()


def main():
    """ [Insert description]
        1. Tries to load numpy arrays from saved files. If not found, calls load_save_data from OceanFlow_utils
        2. Calls the ocean_streamplot function to create and save a streamplot of the flow data"""
    try:
        uv_mask_data = np.load("u_3d.npy"), np.load("v_3d.npy"), np.load("mask.npy").astype('bool')
    except FileNotFoundError:
        OceanFlow_utils.load_save_data()
        uv_mask_data= np.load("u_3d.npy"), np.load("v_3d.npy"), np.load("mask.npy").astype('bool')

    u_3d, v_3d, mask = uv_mask_data
    size_extent_dict = OceanFlow_utils.get_3d_size_extent(u_3d)
    rows = size_extent_dict["rows"]
    columns = size_extent_dict["columns"]

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
    # TODO Commenting this out for now because it takes a long time to run. Uncomment for final version ###
    plane_crash_coordinates = [400, 400]
    plane_crash(*uv_mask_data, plane_crash_coordinates)

    # -------------------------------------------------------------------------------------------------------------#


    # x_start = 50
    # y_start = 300
    # u = u_array[x_start, y_start, :]  # 100 x 1 array. Need covariance of each point --> 100 x 100 array
    # v = v_array[x_start, y_start, :]
    # l2 = 10
    # sig2 = .001
    # tau = .0001


if __name__ == "__main__":
    main()
