import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sklearn.mixture
from scipy.stats import norm
from tensorflow import keras
import tensorflow as tf

plot_path = '/mnt/c/Users/schuy/Documents/py/plots/baby_bayes'

def plt_img(img_array, label, fname, plot_path=plot_path):
    fig, ax = plt.subplots()
    ax.imshow(img_array)
    ax.set_title(label)
    fout = os.path.join(plot_path, fname)
    fig.savefig(fout)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

N, x_dim, y_dim = x_train.shape[0], x_train.shape[1], x_train.shape[2]

gm = sklearn.mixture.GaussianMixture(n_components=50, covariance_type='diag', init_params='kmeans').fit(x_train.reshape(N, x_dim*y_dim))
means = {component:mean for component,mean in zip(range(10),gm.means_)}
covs = {comp:cov for comp,cov in zip(range(10), gm.covariances_)}


for key in list(means.keys()):
    mu = means[key].reshape(28,28)
    cov = covs[key].reshape(28,28)
    img = norm.pdf(mu, cov)


    fig,ax = plt.subplots()
    ax.imshow(img)
    fig.colorbar(cm.ScalarMappable(), ax=ax)
    fout = os.path.join(plot_path, 'GMM_plots/k_means_init/component_{}.png'.format(key))
    if not os.path.exists(os.path.dirname(fout)):
        os.makedirs(os.path.dirname(fout))
    fig.savefig(fout)

