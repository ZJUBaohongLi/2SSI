import numpy as np
from config import Config
from torchvision.datasets import MNIST
from ..data.data_class import TrainDataSet, TestDataSet

X_mnist = None
y_mnist = None


def sensf(x):
    return 2.0 * ((x - 5) ** 4 / 600 + np.exp(-((x - 5) / 0.5) ** 2) + x / 10. - 2)


def emocoef(emo):
    emoc = (emo * np.array([1., 2., 3., 4., 5., 6., 7.])[None, :]).sum(axis=1)
    return emoc


psd = 3.7
pmu = 17.779
ysd = 158.  # 292.
ymu = -292.1


def storeg(x, price):
    emoc = emocoef(x[:, 1:])
    time = x[:, 0]
    g = sensf(time) * emoc * 10. + (emoc * sensf(time) - 2.0) * (psd * price.flatten() + pmu)
    y = (g - ymu) / ysd
    return y.reshape(-1, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loadmnist():
    '''
    Load the mnist data once into global variables X_mnist and y_mnist.
    '''
    global X_mnist
    global y_mnist
    train, test = MNIST('data/', True), MNIST('data/', False)
    X_mnist = []
    y_mnist = []
    for d in [train, test]:
        X, y = d.data.numpy(), d.targets.numpy()
        X = X.astype('float32')
        X /= 255.
        idx = np.argsort(y)
        X_mnist.append(X[idx, :, :])
        y_mnist.append(y[idx])


def get_images(digit, n, seed=None, testset=False):
    if X_mnist is None:
        loadmnist()
    is_test = int(testset)
    rng = np.random.RandomState(seed)
    X_i = X_mnist[is_test][y_mnist[is_test] == digit, :, :]
    n_i, i, j = X_i.shape
    perm = rng.permutation(np.arange(n_i))
    if n > n_i:
        raise ValueError('You requested %d images of digit %d when there are \
						  only %d unique images in the %s set.' % (n, digit, n_i, 'test' if testset else 'training'))
    return X_i[perm[0:n], :, :].reshape((n, i * j))


def selection_rule(cause):
    b = np.random.randn(len(cause), 1)
    w = np.array(
        [-1.0] + [0.1] * (cause.shape[1] - 2) + [1 * Config.c_strength]
    ).reshape((cause.shape[1], 1))
    p = np.dot(cause, w) + b
    p = sigmoid(p)
    selection_res = np.ones(p.shape)
    for i in range(len(selection_res)):
        selection_res[i, 0] = np.random.binomial(1, p[i, 0], 1)
    return selection_res


def demand(n, seed=1, ynoise=1., pnoise=1., ypcor=0.8, use_images=False, test=False):
    rng = np.random.RandomState(seed)

    # covariates: time and emotion
    time = rng.rand(n) * 10
    emotion_id = rng.randint(0, 7, size=n)
    # emotion = one_hot(emotion_id)
    emotion = emotion_id.reshape((-1, 1))
    if use_images:
        idx = np.argsort(emotion_id)
        emotion_feature = np.zeros((0, 28 * 28))
        for i in range(7):
            img = get_images(i, np.sum(emotion_id == i), seed, test)
            emotion_feature = np.vstack([emotion_feature, img])
        reorder = np.argsort(idx)
        emotion_feature = emotion_feature[reorder, :]
    else:
        emotion_feature = emotion

    # random instrument
    z = rng.randn(n)

    # erros
    e = rng.randn(n)

    # z, u -> price
    v = rng.randn(n) * pnoise
    price = sensf(time) * (z + 3) + 25.
    price = price + v
    price = (price - pmu) / psd + Config.u_strength * e
    price = price.reshape((-1, 1))
    # true observable demand function
    x = np.concatenate([time.reshape((-1, 1)), emotion_feature], axis=1)
    x_latent = np.concatenate([time.reshape((-1, 1)), emotion], axis=1)
    g = lambda x, z, p: storeg(x, p)  # doesn't use z

    # errors
    # e = (ypcor * ynoise / pnoise) * v + rng.randn(n) * ynoise * np.sqrt(1 - ypcor ** 2)
    e = e.reshape(-1, 1)

    # response
    y = g(x_latent, None, price) + e
    gt = g(x_latent, None, price)

    z = np.concatenate([z.reshape((-1, 1)), x], axis=1)

    s = selection_rule(np.concatenate([price, z, y], axis=1))
    selected_index = np.where(s.reshape(-1) == 1)[0]
    new_selected_index = np.random.permutation(selected_index)
    new_t = price[new_selected_index]
    new_x = x[new_selected_index]
    new_z = z[new_selected_index]
    new_y = y[new_selected_index]
    new_s = s[new_selected_index]
    new_gt = gt[new_selected_index]
    unselected_index = np.where(s.reshape(-1) == 0)[0]
    new_unselected_index = np.random.permutation(unselected_index)
    unselected_new_t = price[new_unselected_index]
    unselected_new_x = x[new_unselected_index]
    unselected_new_z = z[new_unselected_index]
    unselected_new_y = y[new_unselected_index]
    unselected_new_s = s[new_unselected_index]
    unselected_new_gt = gt[new_unselected_index]

    return TrainDataSet(treatment=new_t[:Config.sample_num * 8 // 10],
                        instrumental=new_z[:Config.sample_num * 8 // 10],
                        covariate=new_x[:Config.sample_num * 8 // 10],
                        outcome=new_y[:Config.sample_num * 8 // 10],
                        structural=new_gt[:Config.sample_num * 8 // 10],
                        selection=new_s[:Config.sample_num * 8 // 10]
                        ), \
           TrainDataSet(treatment=unselected_new_t[:Config.sample_num * 8 // 10],
                        instrumental=unselected_new_z[:Config.sample_num * 8 // 10],
                        covariate=unselected_new_x[:Config.sample_num * 8 // 10],
                        outcome=unselected_new_y[:Config.sample_num * 8 // 10],
                        structural=unselected_new_gt[:Config.sample_num * 8 // 10],
                        selection=unselected_new_s[:Config.sample_num * 8 // 10]
                        ), \
           TestDataSet(
               treatment=new_t[Config.sample_num * 8 // 10:Config.sample_num],
               instrumental=new_z[Config.sample_num * 8 // 10:Config.sample_num],
               covariate=new_x[Config.sample_num * 8 // 10:Config.sample_num],
               structural=new_gt[Config.sample_num * 8 // 10:Config.sample_num]
           ), \
           TestDataSet(
               treatment=unselected_new_t[Config.sample_num * 8 // 10:Config.sample_num],
               instrumental=unselected_new_z[Config.sample_num * 8 // 10:Config.sample_num],
               covariate=unselected_new_x[Config.sample_num * 8 // 10:Config.sample_num],
               structural=unselected_new_gt[Config.sample_num * 8 // 10:Config.sample_num]
           )
