# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:20:37 2022

@author: Theo.ROSSI
"""


from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.05, 0.5, 20)},
                    cv=20) # 20-fold cross-validation
grid.fit(np.array(AMP1_4)[:, None])
print (grid.best_params_)


x_grid = np.linspace(np.min(var), np.max(var), len(var))  #var: variable to use in the histogram

ax_hist.plot(x_grid, kde_sklearn(np.array(var), x_grid, bandwidth=0.1), linewidth=3, alpha=0.5)



# def gauss_func(mu,sigma,bins):
#     '''
#     mu : mean
#     sigma : standard deviation
#     bins : histogram bins

#     Returns
#     -------
#     y : gaussian function of a histogram dataset

#     '''
#     y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) 
#     return y

# his_15, bin_edges_15 = np.histogram(AMP1_15, bins=20, density=True)
# his_4, bin_edges_4 = np.histogram(AMP1_4, bins=20, density=True)
# ax_hist.plot(bin_edges_15, gauss_func(np.mean(AMP1_15), np.std(AMP1_15), bin_edges_15), 'k')
# ax_hist.plot(bin_edges_4, gauss_func(np.mean(AMP1_4), np.std(AMP1_4), bin_edges_4), 'r')