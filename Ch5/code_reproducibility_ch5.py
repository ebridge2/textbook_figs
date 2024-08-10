#%%  Section 5.5 (already did the others in the old one that I lost)

from graspologic.simulations import sbm
import numpy as np
from sklearn.preprocessing import LabelEncoder

n = 100  # the number of nodes
M = 8  # the total number of networks
# human brains have homophilic block structure
Bhuman = np.array([[0.2, 0.02], [0.02, 0.2]])
# alien brains have a core-periphery block structure
Balien = np.array([[0.4, 0.2], [0.2, 0.1]])

# set seed for reproducibility
np.random.rand(123)
# generate 4 human and 4 alien brain networks
A_humans = [sbm([n // 2, n // 2], Bhuman) for i in range(M // 2)]
A_aliens = [sbm([n // 2, n // 2], Balien) for i in range(M // 2)]
# concatenate list of human and alien networks
networks = A_humans + A_aliens

# 1 = left hemisphere, 2 = right hemisphere for node communities
le = LabelEncoder()
labels = np.repeat(["L", "R"], n // 2)
zs = le.fit_transform(labels) + 1
# %%
from graspologic.embed import AdjacencySpectralEmbed as ase

# embed the first network
Xhat = ase(n_components=2, svd_seed=123).fit_transform(A_humans[0])
# %%
# a rotation by 90 degrees
W = np.array([[0, -1], [1, 0]])
Yhat = Xhat @ W
# %%
# check that probability matrix is the same
np.allclose(Yhat @ Yhat.transpose(), Xhat @ Xhat.transpose())
# returns True
# %%
# embed the second network
Xhat2 = ase(n_components=2).fit_transform(networks[1])
# %%
# embed the first alien network
Xhat_alien = ase(n_components=2, svd_seed=123).fit_transform(A_aliens[0])

# compute frob norm between first human and third human net
# estimated latent positions
dist_firsthum_thirdhum = np.linalg.norm(Xhat - Xhat2, ord="fro")
print("Frob. norm(first human, third human) = {:3f}".format(dist_firsthum_thirdhum))
# Frob. norm(first human, third human) = 8.738087

# compute frob norm between first human and first alien net
# estimated latent positions
dist_firsthum_alien = np.linalg.norm(Xhat - Xhat_alien, ord="fro")
print("Frob. norm(first human, alien) = {:3f}".format(dist_firsthum_alien))
# Frob. norm(first human, alien) = 5.220325
# %%
from graspologic.embed import MultipleASE as mase

# Use mase to embed everything
mase = mase(n_components=2, svd_seed=123)
# fit_transform on the human and alien networks simultaneously
# + combines the two lists
latents_mase = mase.fit_transform(networks)
# %%
from graspologic.embed import AdjacencySpectralEmbed as ase

dhat = int(np.ceil(np.log2(n)))
# spectrally embed each network into ceil(log2(n)) dimensions with ASE
separate_embeddings = [
    ase(n_components=dhat, svd_seed=123).fit_transform(network) for network in networks
]
# %%
# Concatenate your embeddings horizontally into a single n x Md matrix
joint_matrix = np.hstack(separate_embeddings)
# %%
def unscaled_embed(X, d):
    U, s, Vt = np.linalg.svd(X)
    return U[:, 0:d]


Vhat = unscaled_embed(joint_matrix, 2)
# %%
# stack the networks into a numpy array
As_ar = np.asarray(networks)
# compute the scores
scores = Vhat.T @ As_ar @ Vhat
# %%
from graphbook_code import generate_sbm_pmtx

Phum = generate_sbm_pmtx(zs, Bhuman)
Palien = generate_sbm_pmtx(zs, Balien)
Pests = Vhat @ scores @ Vhat.T
# %%
from graspologic.embed import MultipleASE as mase

d = 2
mase_embedder = mase(n_components=d)
# obtain an estimate of the shared latent positions
Vhat = mase_embedder.fit_transform(networks)
# obtain an estimate of the scores
Rhat_hum1 = mase_embedder.scores_[0]
# obtain an estimate of the probability matrix for the first human
Phat_hum1 = Vhat @ mase_embedder.scores_[0] @ Vhat.T
# %%
omni_ex = np.block(
    [
        [networks[0], (networks[0] + networks[1]) / 2],
        [(networks[1] + networks[0]) / 2, networks[1]],
    ]
)
# %%
from graspologic.embed.omni import _get_omni_matrix

omni_mtx = _get_omni_matrix(networks)
# %%
from graspologic.embed import AdjacencySpectralEmbed as ase

dhat = int(np.ceil(np.log2(n)))
Xhat_omni = ase(n_components=dhat).fit_transform(omni_mtx)
# %%
M = len(networks)
n = len(networks[0])

# obtain a M x n x d tensor
Xhat_tensor = Xhat_omni.reshape(M, n, -1)
# the estimated latent positions for the first network
Xhat_human1 = Xhat_tensor[0, :, :]
# %%
from graspologic.embed import OmnibusEmbed as omni

# obtain a tensor of the estimated latent positions
Xhat_tensor = omni(n_components=int(np.log2(n))).fit_transform(networks)
# obtain the estimated latent positions for the first human
# network
Xhat_human1 = Xhat_tensor[0, :, :]
# %%
Phat_hum1 = Xhat_human1 @ Xhat_human1.T
# %% # Section 5.6
from graphbook_code import heatmap
# %%
from graspologic.simulations import sbm
import numpy as np

n = 200  # total number of nodes
# first two communities are the ``core'' pages for statistics
# and computer science, and second two are the ``peripheral'' pages
# for statistics and computer science.
B = np.array([[.4, .3, .05, .05],
              [.3, .4, .05, .05],
              [.05, .05, .05, .02],
              [.05, .05, .02, .05]])

# make the stochastic block model
A, labels = sbm([n // 4, n // 4, n // 4, n // 4], B, return_labels=True)
# generate labels for core/periphery
co_per_labels = np.repeat(["Core", "Periphery"], repeats=n//2)
# generate labels for statistics/CS.
st_cs_labels = np.repeat(["Stat", "CS", "Stat", "CS"], repeats=n//4)
# %%
trial = []
for label in st_cs_labels:
    if "Stat" in label:
        # if the page is a statistics page, there is a 50% chance
        # of citing each of the scholars
        trial.append(np.random.binomial(1, 0.5, size=20))
    else:
        # if the page is a CS page, there is a 5% chance of citing
        # each of the scholars
        trial.append(np.random.binomial(1, 0.05, size=20))
Y = np.vstack(trial)
# %%
def embed(X, d=2):
    """
    A function to embed a matrix.
    """
    Lambda, V = np.linalg.eig(X)
    return V[:, 0:d] @ np.diag(np.sqrt(np.abs(Lambda[0:d])))

def pca(X, d=2):
    """
    A function to perform a pca on a data matrix.
    """
    X_centered = X - np.mean(X, axis=0)
    return embed(X_centered @ X_centered.T, d=d)

Y_embedded = pca(Y, d=2)
# %%
from graspologic.utils import to_laplacian

# compute the network Laplacian
L_wiki = to_laplacian(A, form="DAD")
# log transform, strictly for visualization purposes
L_wiki_logxfm = np.log(L_wiki + np.min(L_wiki[L_wiki > 0])/np.exp(1))

# compute the node similarity matrix
Y_sim = Y @ Y.T
# %%
from graspologic.embed import AdjacencySpectralEmbed as ase

def case(A, Y, weight=0, d=2, tau=0):
    """
    A function for performing case.
    """
    # compute the laplacian
    L = to_laplacian(A, form="R-DAD", regularizer=tau)
    YYt = Y @ Y.T
    return ase(n_components=2).fit_transform(L + weight*YYt)

embedded = case(A, Y, weight=.002)
# %%
from graspologic.embed import CovariateAssistedEmbed as case

embedding = case(alpha=None, n_components=2).fit_transform(A, covariates=Y)
# %%
embedding = case(assortative=False, n_components=2).fit_transform(A, covariates=Y)
# %%  # Section 5.7
