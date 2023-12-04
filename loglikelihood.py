import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from scipy.special import factorial as fact

#x = np.random.normal(0, 1, 1000)
#norm = lambda x, mu, sig: (2*np.pi*sig**2)**-.5*np.exp(-0.5*((x-mu)/sig)**2)
#x = np.linspace(-2,2,1000)

fact_choice = lambda n, x: fact(n)/(fact(n-x)*fact(x)) 
bin = lambda x, p, n: fact_choice(n, x)*p**x*(1-p)**(n-x)

X = np.random.binomial(5,0.3,20)

P = np.linspace(0,1,20)
n = 5

pdfs_X = [bin(X, p, n) for p in P]

L_X = np.prod(pdfs_X, axis=1)
for row, p in zip(L_X, P):
    print(row, p)


'''
L_x = np.prod(sample)
log_L_x = np.log(L_x)

#print(norm_x.shape, norm_x[:10])

print('total_likelihood:', L_x)
print('log_likelihood:', log_L_x)

'''

'''
fig, ax = plt.subplots()
#ax.hist(norm(x, 0, 1), bins=100)
#ax.hist(norm.pdf(x))
ax.plot(x, norm.pdf(x))

path = os.path.join(os.getcwd(), 'normal_x_selfgen.png')
fig.savefig(path)
'''