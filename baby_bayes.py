import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats
from tensorflow import keras
import tensorflow as tf

plot_path = '/mnt/c/Users/schuy/Documents/py/plots/baby_bayes'

def plt_img(img_array, label, fname, plot_path=plot_path):
    fig, ax = plt.subplots()
    ax.imshow(img_array)
    ax.set_title('Digit class y_k=({}): using y_k mu_k,var_k \n parameters to sample from N(mu_k,var_k)'.format(label))
    fout = os.path.join(plot_path, fname)
    fig.savefig(fout)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

## Segregate all instances of all classes within a sample; \{X\}_{y_i} for all y_i
y_train_class_indices = {}
x_train_classes = {}

unique_classes = np.sort(np.unique(y_train))
for class_num in unique_classes:
    y_train_class_indices[class_num] = np.where(y_train == class_num)[0]

for class_num in unique_classes:
    indices = y_train_class_indices[class_num]
    x_train_classes[class_num] = x_train[indices]

## Generate distributions
## For now, taking global mean over all pixel values
mu_classes = {}
var_classes = {}
for class_num in unique_classes:
    x_data = x_train_classes[class_num]
    x_data_flat = np.vstack([x.flatten() for x in x_data])
    mu_classes[class_num] = np.mean(x_data_flat, axis=0)
    var_classes[class_num] = np.var(x_data_flat, axis=0)

## Sample random class Gaussian pdf, plot image
## this is solely modeling p(X|Y)
p_X_Yk = {}

for class_num in unique_classes:
    mu_k = mu_classes[class_num]
    var_k = var_classes[class_num]
    p_X_Yk[class_num] = np.random.normal(mu_k, var_k**.5).reshape(28,28)
    
#plt_img(p_x_y.reshape(28,28),'{}'.format(selected_class), 'normal_sample_class_{}.png'.format(selected_class))

## Generate posterior probability model
## Extending above to p(y|x) = p(x|y)p(y)/p(x)
binned_X = np.around(x_data)
X_bins = np.sort(np.unique(binned_X))
p_X = {bin:len(np.where(binned_X == bin)[0])/(x_data.shape[0]*x_data.shape[1]*x_data.shape[2]) for bin in X_bins}

p_Y = {}
for class_num in unique_classes:
    p_Y[class_num] = len(y_train_class_indices[class_num])/len(y_train)

### Now we have full expression for p(Y|X), could test prediction abilities w simple Gaussian model

