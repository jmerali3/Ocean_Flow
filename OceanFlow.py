import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import matplotlib.animation as animation
import seaborn as sns
from sklearn.model_selection import KFold

# (x, y, T) file
u_array = np.load("u_array.npy")
v_array = np.load("v_array.npy")
rng = default_rng(12345)

x_start = 50
y_start = 300
u = u_array[x_start, y_start, :]  # 100 x 1 array. Need covariance of each point --> 100 x 100 array
v = v_array[x_start, y_start, :]
l2 = 10
sig2 = .001
tau = .0001



def compute_kernel(arr, l_sq, sig_sq, arr2 = None):
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
    L = np.linalg.cholesky(K_ss + tau*np.eye(n))
    f_prior = np.dot(L, np.random.normal(size=(n, 3)))

    K = compute_kernel(train, l2, sig2)
    L = np.linalg.cholesky(K + tau*np.eye(len(train)))

    K_s = compute_kernel(train, l2, sig2, test)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, arr)).reshape((n,)) # This is where labels come in (arr)

    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)

    L_final = np.linalg.cholesky(K_ss + tau*np.eye(n) - np.dot(Lk.T, Lk))
    f_post = mu.reshape(-1,1) + np.dot(L_final, np.random.normal(size = (n,1))) # mu + beta*N(0,I), draws random samples
    if plot:
        plt.plot(train, arr, c='crimson', lw=3, label="Ground Truth")
        plt.plot(test, f_post, c='darkgreen', label="Predicted F_Posterior")
        plt.gca().fill_between(test.flat, mu-3*stdv, mu+3*stdv, color="#dddddd")
        plt.plot(test, mu, 'r--', lw=3, label="Mu - Average", c="darkblue")
        title = "U Direction" if arr is u else "V Direction Interpolation"
        plt.title(title +f" tau = {tau}, l^2 = {l2}, sig^2 = {sig2}")
        plt.legend()
        plt.show()

    return f_post, mu, stdv

#f_post, mu, stdv = interpolate(u, 300, l2, sig2, tau)


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
        x_current = x_current + u[i]*24   # km = km + km/hr * 24 hr/day. Each time step is 1 day in f_post
        y_current = y_current + v[i]*24

    #movement_summary = pd.DataFrame({'x': x, 'y': y, 'u': u, 'v': v}) # row is timestep, columns are x coord, y coord, x magnitude, y magnitude
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
        L = np.linalg.cholesky(K_train + tau*np.eye(len(train)))
        Lk = np.linalg.solve(L, K_cross)
        mu = np.dot(Lk.T, np.linalg.solve(L, train_labels))
        stdv = np.sqrt(np.diag(K_test) - np.sum(Lk**2, axis=0))
        L_final = np.linalg.cholesky(K_test + tau*np.eye(n) - np.dot(Lk.T, Lk))
        f_post = mu.reshape(-1,1) + L_final @ np.random.normal(size=(n,1))
        log_lik = -.5*f_post.T @ np.linalg.solve(K_test, f_post) - .5 * np.log(np.linalg.det(K_test)) - 10/2 * np.log(2*np.pi)
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

    return loss, round(np.array(loss).mean(),2) #, fpost_list, stdv_list, mu_list, test_list

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


def streamplots():
    x = np.arange(504)
    y = np.arange(555)
    fig, ax = plt.subplots()
    strm = ax.streamplot(x, y, u_array[:,:,0], v_array[:,:,0], linewidth=2, cmap=plt.cm.autumn)
    fig.colorbar(strm.lines)
    plt.show()


def compute_movement(x_current, y_current):
    # computes new movements and locations for each time step, given x and y initial coordinates
    x, y, u, v = [np.zeros(100) for _ in range(4)]
    for i in range(100):
        x[i] = x_current
        y[i] = y_current
        u[i] = u_array[int(round(x_current, 0)), int(round(y_current, 0)), i]
        v[i] = v_array[int(round(x_current, 0)), int(round(y_current, 0)), i]
        x_current = x_current + u[i]   # Time step is 3 hours and each grid space is 3 hours = 1 grid space per time unit
        y_current = y_current + v[i]

    #movement_summary = pd.DataFrame({'x': x, 'y': y, 'u': u, 'v': v}) # row is timestep, columns are x coord, y coord, x magnitude, y magnitude
    movement_summary = np.stack((x, y, u, v), axis=1)
    return movement_summary


def get_coordinates(length):
    coordinates = []
    for i in range(length):
        x = rng.integers(low=0, high=554, endpoint=True)
        y = rng.integers(low=0, high=503, endpoint=True)
        if u_array[x, y, 0] != 0 and v_array[x, y, 0] != 0:
            coordinates.append((x, y))
    return coordinates


def get_coordinates_toy(length, mu_x, mu_y, var_x, var_y):
    x = rng.normal(mu_x, var_x, length-1)
    x = np.append(x, mu_x)
    y = rng.normal(mu_y, var_y, length-1)
    y = np.append(y, mu_y)
    coordinates = list(zip(x, y))
    return coordinates


def animate(n):
    def get_colors(length):
        rbg = []
        for i in range(length):
            rbg.append(tuple(rng.random(3,)))
        return rbg

    sig_x = 30
    sig_y = 30
    #coordinates = get_coordinates_toy(10, 100, 350, sig_x, sig_y) # 10 is the number of samples
    # Can use either get_coordinates or get_coordinates_toy depending on what you want
    coordinates = get_coordinates(50)
    colors = get_colors(len(coordinates))

    tx, ty = [np.zeros([n, len(coordinates)]) for _ in range(2)]

    for i, (x, y) in enumerate(coordinates):
        movement = compute_movement_extended(x, y, n) # n is number of days
        tx[:, i] = movement[:, 0]
        ty[:, i] = movement[:, 1]

    fig, ax = plt.subplots()
    # ax.set_xlim([0, 554])
    # ax.set_ylim([0, 503])
    ax.set_xlim([0, 250])
    ax.set_ylim([250, 503])
    ax.scatter([],[])

    def animate2(i):
        ax.set_title(f"T = {i+1}, var_x = {sig_x}, var_y = {sig_y}")
        ax.scatter(tx[i, :], ty[i, :], c=colors, s=2)
        return ax

    ani = animation.FuncAnimation(fig, animate2, frames=n, interval=50, repeat=False)
    ani.save("ToySearchDaysMany2.gif", writer=animation.PillowWriter(fps=20))
    plt.show()

animate(n=300)

def compute_correlations(ar1, ar2):
    # ar is a 1D array (1x100) for point x,y for times T = 1 - 100
    if np.array_equal(ar1, ar2):
        return 0
    else:
        return (np.corrcoef(ar1, ar2)[0, 1]).round(2)


def get_corr_dict(arr1, arr2):
    samples = range(100000)
    correlations_dict = {}
    for _ in samples:
        x_rand1 = rng.integers(low=0, high=554, endpoint=True)
        x_rand2 = rng.integers(low=0, high=554, endpoint=True)
        y_rand1 = rng.integers(low=0, high=503, endpoint=True)
        y_rand2 = rng.integers(low=0, high=503, endpoint=True)
        point_vec_A = arr1[x_rand1, y_rand1, :]
        point_vec_B = arr1[x_rand2, y_rand2, :]

        if point_vec_A.all() == 0 or point_vec_B.all() == 0:
            continue

        corr1 = compute_correlations(point_vec_A, point_vec_B)

        if abs(corr1) > .9 and (x_rand1-x_rand2) > 5 and (y_rand1-y_rand2) > 5:
            point_vec_C = arr2[x_rand1, y_rand1, :]
            point_vec_D = arr2[x_rand2, y_rand2, :]
            corr2 = compute_correlations(point_vec_C, point_vec_D)

            if abs(corr2) > .9:
                key = (x_rand1, y_rand1, x_rand2, y_rand2)
                correlations_dict[key] = (corr1, corr2)
    return correlations_dict

#correlations = get_corr_dict(u_array, v_array)

# for key, value in correlations.items():
#     x1, y1, x2, y2 = key
#     corr1, corr2 = value
#     print(f"Point 1: ({x1},{y1}); Point 2: ({x2},{y2}); u-corr: {corr1}; v-corr: {corr2}")



# print(u_array[358, 87, :])
# print(u_array[98, 6, :])
# print(v_array[358, 87, :])
# print(v_array[98, 6, :])


def other_calcs():
    velocity = np.load("velocity.npy")
    velocity[velocity == 0] = np.nan
    var_velocity = np.nanvar(velocity, axis=2)
    var_velocity[var_velocity == 0] = np.nan
    min_var =np.nanmin(var_velocity)
    min_var_loc = np.where(var_velocity == min_var)
    max_x = np.max(u_array)
    max_x_loc = np.where(u_array == max_x)
    x_avg = np.mean(u_array.flatten())
    y_avg = np.mean(v_array.flatten())
    print(velocity[358, 87, :].mean())
    print(velocity[98, 6, :].mean())


### CALC VELOCITY ###
def calc_velocity():
    velocity = np.sqrt(np.square(u_array) + np.square(v_array))
    print(velocity.shape)
    np.save("velocity", velocity, allow_pickle=True)


### LOAD DATA ###
def load_save_data():
    time = range(2, 101)
    u_array = np.transpose(np.loadtxt("OceanFlow/1u.csv", delimiter=','))
    v_array = np.transpose(np.loadtxt("OceanFlow/1v.csv", delimiter=','))

    for i in time:
        u_array_open = np.transpose(np.loadtxt("OceanFlow/" + str(i) + "u.csv", delimiter=','))
        u_array = np.dstack((u_array, u_array_open))
        v_array_open = np.transpose(np.loadtxt("OceanFlow/" + str(i) + "v.csv", delimiter=','))
        v_array = np.dstack((v_array, v_array_open))

    np.save("u_array", u_array, allow_pickle=True)
    np.save("v_array", v_array, allow_pickle=True)


