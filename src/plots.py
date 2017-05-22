import matplotlib.pyplot as plt
from itertools import combinations

def plot_pca(Y_pca,y,labels):
    for combination in combinations(range(Y_pca.shape[1]),2):
        for l in labels:
            plt.plot(Y_pca[y==l,combination[0]],Y_pca[y==l,combination[1]],'o')
        plt.show()
