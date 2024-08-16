def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path, mode=0o777)

def load_agw_1d(base_dir, get_feats=False):
    if not path.exists(base_dir + '/agw_data'):
        mkdir(base_dir + '/agw_data')
        urllib.urlretrieve('https://raw.githubusercontent.com/wjmaddox/drbayes/master/experiments/synthetic_regression/ckpts/data.npy',
                           filename=base_dir + '/agw_data/data.npy')

    def features(x):
        return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])

    data = np.load(base_dir + '/agw_data/data.npy')
    x, y = data[:, 0], data[:, 1]
    y = y[:, None]
    f = features(x)

    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)
    f_means, f_stds = f.mean(axis=0), f.std(axis=0)

    X = ((x - x_means) / x_stds).astype(np.float32)
    Y = ((y - y_means) / y_stds).astype(np.float32)
    F = ((f - f_means) / f_stds).astype(np.float32)

    if get_feats:
        return F, Y

    return X[:, None], Y

#### Note the location the data will be stored!!!

folder = '../../../data/'
X, Y = load_agw_1d(folder, get_feats=False)