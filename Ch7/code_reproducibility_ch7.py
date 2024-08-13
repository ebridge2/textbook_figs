# %%  Section 7.1 two-sample testing for networks

from graspologic.simulations import sbm
import numpy as np
from graphbook_code import dcsbm, generate_dcsbm_pmtx, generate_sbm_pmtx

np.random.seed(0)

n = 100  # the number of nodes
# human brains have homophilic block structure
Bhum = np.array([[0.2, 0.02], [0.02, 0.2]])
# alien brains add degree-correction
theta_alien = np.tile(np.linspace(1.5, 0.5, n // 2), 2)

# generate human and alien brain network
A_human, z = sbm([n // 2, n // 2], Bhum, return_labels=True)
A_alien = dcsbm(z, theta_alien, Bhum)

Phum = generate_sbm_pmtx(z, Bhum)
Palien = generate_dcsbm_pmtx(z, theta_alien, Bhum)
# %%
from scipy.linalg import orthogonal_procrustes
from graspologic.embed import AdjacencySpectralEmbed as ase

d = 2
# estimate latent positions for alien and human networks
Xhat_human = ase(n_components=d).fit_transform(A_human)
Xhat_alien = ase(n_components=d).fit_transform(A_alien)
# estimate best possible rotation of Xhat_alien to Xhat_human by
# solving orthogonal procrustes problem
W = orthogonal_procrustes(Xhat_alien, Xhat_human)[0]
observed_norm = np.linalg.norm(Xhat_human - Xhat_alien @ W, ord="fro")
# %%
from graspologic.simulations import rdpg


def generate_synthetic_networks(X):
    """
    A function which generates two synthetic networks with
    same latent position matrix X.
    """
    A1 = rdpg(X, directed=False, loops=False)
    A2 = rdpg(X, directed=False, loops=False)
    return A1, A2


Ap, App = generate_synthetic_networks(Xhat_human)


# %%
def compute_latent(A, d):
    """
    A function which returns the latent position estimate
    for an adjacency matrix A.
    """
    return ase(n_components=d).fit_transform(A)


Xhat_p = compute_latent(Ap, d)
Xhat_pp = compute_latent(App, d)


# %%
def compute_norm_orth_proc(A, B):
    """
    A function which finds the best rotation of B onto A,
    and then computes and returns the norm.
    """
    R = orthogonal_procrustes(B, A)[0]
    return np.linalg.norm(A - B @ R)


norm_null = compute_norm_orth_proc(Xhat_p, Xhat_pp)


# %%
def parametric_resample(A1, A2, d, nreps=100):
    """
    A function to generate samples of the null distribution under H0
    using parametric resampling.
    """
    null_norms = np.zeros(nreps)
    Xhat1 = compute_latent(A1, d)
    for i in range(0, nreps):
        Ap, App = generate_synthetic_networks(Xhat1)
        Xhat_p = compute_latent(Ap, d)
        Xhat_pp = compute_latent(App, d)
        null_norms[i] = compute_norm_orth_proc(Xhat_p, Xhat_pp)
    return null_norms


nreps = 100
null_norms = parametric_resample(A_alien, A_human, 2, nreps=nreps)
# %%
pval = ((null_norms >= observed_norm).sum() + 1) / (nreps + 1)
print(f"estimate of p-value: {pval:.5f}")
# estimate of p-value: 0.00990
# %%
from graspologic.inference import latent_position_test

nreps = 100  # the number of null replicates
lpt = latent_position_test(
    A_human, A_alien, n_bootstraps=nreps, n_components=d, workers=-1
)
print("estimate of p-value: {:.5f}".format(lpt[1]))
# estimate of p-value: 0.00990
# %%
# generate a new human brain network with same block matrix
A_human2 = sbm([n // 2, n // 2], Bhum)

lpt_hum2hum = latent_position_test(
    A_human, A_human2, n_bootstraps=nreps, n_components=d, workers=-1
)
print("estimate of p-value: {:.5f}".format(lpt_hum2hum[1]))
# estimate of p-value: 0.386
# %%
ncoins = 300  # the number of coins in each container

# the probabilities from container 1 landing on heads
# with a much larger variance
pi1 = np.random.beta(a=4, b=4, size=ncoins)

# the probabilities of container 2 landing on heads,
# with a much smaller variance
pi2 = np.random.beta(a=15, b=15, size=ncoins)
# %%
from graspologic.inference import latent_distribution_test

nreps = 100
approach = "mgc"  # the strategy for the latent distribution test
ldt_dcorr = latent_distribution_test(
    A_human, A_alien, test=approach, metric="euclidean", n_bootstraps=nreps, workers=-1
)
print("estimate of p-value: {:.5f}".format(ldt_dcorr.pvalue))
# %%
print("estimate of p-value: {:.4f}".format(ldt_dcorr[1]))
# %%  Section 7.2 Two-sample stesting for SBMs

import numpy as np
from graspologic.simulations import sbm

np.random.seed(0)

ns = [45, 30, 25]  # number of towns

states = ["NY", "NJ", "PA"]
# z is a column vector indicating which state each town is in
z = np.repeat(states, ns)

Bnight = np.array([[0.3, 0.2, 0.1], [0.2, 0.3, 0.2], [0.1, 0.2, 0.3]])
Bday = Bnight * 1.5  # day time block matrix is generally 50% more than night

# people tend to commute from New Jersey to New York during the day
Bday[0, 1] = 0.4
Bday[1, 0] = 0.4

Anight = sbm(ns, Bnight)
Aday = sbm(ns, Bday)
# %%
from scipy.stats import fisher_exact

K = 3
Pvals = np.empty((K, K))
# fill matrix with NaNs
Pvals[:] = np.nan

# get the indices of the upper triangle of Aday
upper_tri_idx = np.triu_indices(Aday.shape[0], k=1)
# create a boolean array that is nxn
upper_tri_mask = np.zeros(Aday.shape, dtype=bool)
# set indices which correspond to the upper triangle to True
upper_tri_mask[upper_tri_idx] = True

for k in range(0, K):
    for l in range(k, K):
        comm_mask = np.outer(z == states[k], z == states[l])
        table = [
            [
                Aday[comm_mask & upper_tri_mask].sum(),
                Anight[comm_mask & upper_tri_mask].sum(),
            ],
            [
                (Aday[comm_mask & upper_tri_mask] == 0).sum(),
                (Anight[comm_mask & upper_tri_mask] == 0).sum(),
            ],
        ]
        Pvals[k, l] = fisher_exact(table)[1]
# %%
import numpy as np
from graspologic.simulations import er_np
import seaborn as sns
from scipy.stats import binomtest

ncoins = 5000  # the number of coins
p = 0.5  # the true probability
n = 500  # the number of flips

# the number of heads from each experiment
experiments = np.random.binomial(n, p, size=ncoins)

# perform binomial test to see if the number of heads we obtain supports that the
# true probabiily is 0.5
pvals = [binomtest(nheads_i, n, p=p).pvalue for nheads_i in experiments]
# %%
from statsmodels.stats.multitest import multipletests

alpha = 0.05  # the desired alpha of the test
_, adj_pvals, _, _ = multipletests(pvals, alpha=alpha, method="holm")
# %%
from graspologic.utils import symmetrize

Pvals_adj = multipletests(Pvals.flatten(), method="holm")[1].reshape(K, K)
Pvals_adj = symmetrize(Pvals_adj, method="triu")
# %%
pval_dif = Pvals_adj.min()
print(f"p-value of block matrix difference: {pval_dif:.4f}")
# p-value of block matrix difference: 0.0000
# %%
from graspologic.inference import group_connection_test

stat, pval_diff_rescale, misc = group_connection_test(
    Aday, Anight, labels1=z, labels2=z, density_adjustment=True
)
Pval_adj_rescaled = np.array(misc["corrected_pvalues"])
print(f"p-value of block matrix difference, after rescaling: {pval_diff_rescale:.4f}")
# p-value of block matrix difference: 0.0000
# %%  Section 7.3 graph matching

import numpy as np

A = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

P = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

row_reordering = P.T @ A
# %%
B = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

P = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

column_reordering = B @ P
# %%
C = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])

P = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

row_reordering_C = P.T @ C
row_column_reordering = row_reordering_C @ P
# %%
twitter = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])

facebook_permuted = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])

# the permutation to unshuffle the facebook
# permuted adjacency matrix
Pu = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

fb_unpermuted = Pu.T @ facebook_permuted @ Pu
# %%
def make_random_permutation(n, random_seed=0):
    """
    A function that generates a random permutation matric $P$ for n elements.

    1. Generate indices from 0 to n-1
    2. shuffle those indices
    3. Place 1s in the matrix P at the positions defined by the shuffled indices.
    """
    rng = np.random.default_rng(seed=random_seed)
    starting_indices = np.arange(n)
    destination_indices = rng.permutation(n)
    P = np.zeros(shape=(n, n))
    P[destination_indices, starting_indices] = 1
    return P
# %%
from graspologic.simulations import er_np

n = 12
p = 0.5

np.random.seed(0)
A = er_np(n=n, p=p)
# make a random permutation matrix
P = make_random_permutation(n)
B = P.T @ A @ P
disagreements = np.linalg.norm(A - B) ** 2
# %%
from graspologic.match import graph_match

gmp = graph_match(A, B, n_init=10, rng=0)
# %%
def make_unshuffler(destination_indices):
    """
    A function which creates a permutation matrix P from a given permutation of the nodes.
    """
    n = len(destination_indices)
    Pu = np.zeros((n, n))
    starting_indices = np.arange(n)
    Pu[destination_indices, starting_indices] = 1
    return Pu


Pu = make_unshuffler(gmp.indices_B)
B_unshuffled = Pu.T @ B @ Pu
disagreements = np.linalg.norm(A - B_unshuffled) ** 2
print(f"Disagreements: {int(disagreements):d}")
# Disagreements: 0.0
# %%
def match_ratio(P, Pu):
    n = P.shape[0]  # the number of nodes
    return (np.diag(Pu @ P) == 1).sum() / n


print(f"match ratio: {match_ratio(P, Pu):.3f}")
# match ratio: 1.000
# %%
from graspologic.simulations import sbm_corr

n_per_block = 75
n_blocks = 3
block_members = np.repeat(n_per_block, repeats=n_blocks)
n_nodes = block_members.sum()
rho = 0.5
block_probs = np.array([[0.7, 0.1, 0.4], [0.1, 0.3, 0.1], [0.4, 0.1, 0.7]])

np.random.seed(0)
A1, A2 = sbm_corr(block_members, block_probs, rho)
disagreements = np.linalg.norm(A1 - A2) ** 2
print(f"Disagreements (Unshuffled): {int(disagreements):d}")
# Disagreements (Unshuffled): 8041
# %%
P = make_random_permutation(n_nodes)
A2_shuffle = P.T @ A2 @ P
disagreements_shuffled = np.linalg.norm(A1 - A2_shuffle) ** 2
print(f"Disagreements (Shuffled): {int(disagreements_shuffled):d}")
# Disagreements (Shuffled): 22201
# %%
# fit with A and shuffled B
gm = graph_match(A1, A2_shuffle, rng=0)

# obtain unshuffled version of the shuffled B
P_unshuffle_noseed = make_unshuffler(gm.indices_B)
A2_unshuffle_noseed = P_unshuffle_noseed.T @ A2_shuffle @ P_unshuffle_noseed

# compute the match ratio
match_ratio_noseed = match_ratio(P, P_unshuffle_noseed)
print(f"Match Ratio, no seeds: {match_ratio_noseed:.3f}")
# Match Ratio, no seeds: 0.004

disagreements_noseed = np.linalg.norm(A1 - A2_unshuffle_noseed) ** 2
print(f"Disagreements, no seeds: {int(disagreements_noseed):d}")
# Disagreements, no seeds: 12810
# %%
def gen_seeds(P, n_seeds, random_seed=0):
    """
    A function to generate n_seeds seeds for a pair of matrices A1 and P^TA2P
    which are initially matched, but P has been applied to permute the nodes
    of A2.
    """
    rng = np.random.default_rng(seed=random_seed)
    n = P.shape[0]
    # obtain n_seeds random seeds from 1:n
    seeds = rng.choice(n, size=n_seeds, replace=False)
    # use the permutation matrix to find where each seed was permuted to
    seeds_permuted = [np.where(P[i, :] == 1)[0] for i in seeds]
    return (seeds, seeds_permuted)
# %%
nseeds = 10  # the number of seeds to use
# select ten nodes at random from A which will serve as seeds

# obtain seeds for nodes of A1 with nodes of A2 after shuffling
seedsA1, seedsA2_shuffled = gen_seeds(P, nseeds)

# run SGM with A1 and shuffled A2, but provide the seed nodes from A as ref_seeds
# and the corresponding position of these seed nodes after shuffling as permuted_seeds
sgm = graph_match(A1, A2_shuffle, partial_match=(seedsA1, seedsA2_shuffled), rng=0)
P_unshuffle_seeds = make_unshuffler(sgm.indices_B)
A2_unshuffle_seeds = P_unshuffle_seeds.T @ A2_shuffle @ P_unshuffle_seeds

match_ratio_seeds = match_ratio(P, P_unshuffle_seeds)
print(f"Match Ratio, seeds: {match_ratio_seeds:.3f}")
# Match Ratio with seeds: 1.000

disagreements_seeds = np.linalg.norm(A1 - A2_unshuffle_seeds) ** 2
print(f"Disagreements, seeds: {int(disagreements_seeds):d}")
# Disagreements, seeds: 8041
# %%
from graspologic.utils import remove_vertices

nremove = 25

# nodes to remove from A2
n_nodes_A2_N = n_nodes - nremove * n_blocks
base_range = np.arange(n_per_block - nremove, n_per_block)
block_offsets = np.array([0, 75, 150])

# repeat a base range for each block and add block offsets
nodes_to_remove = np.repeat(base_range, len(block_offsets))
nodes_to_remove += np.tile(block_offsets, nremove)
N = np.setdiff1d(np.arange(n_nodes), nodes_to_remove)

# use the remove_vertices function to compute
# the subnetwork induced by the nodes nodes_to_retain
A2_N = remove_vertices(A2, nodes_to_remove)
# %%
A1_N = remove_vertices(A1, nodes_to_remove)
# %%
A2_N_padded = np.pad(A2_N, pad_width=[(0, nremove * n_blocks), (0, nremove * n_blocks)])
# %%
nseeds_padded = 10

np.random.seed(0)
# obtain which nodes of A2 will be the seeds to use, from the retained nodes in the network
seeds_A2_N = np.random.choice(n_nodes_A2_N, size=nseeds_padded, replace=False)

# obtain the nodes in A1
seeds_A1 = N[seeds_A2_N]

# run SGM with A1 and the padded network A2
# since we didn't shuffle A(2),r, we do not need
# to worry about permuting the seeds
sgm_naive = graph_match(
    A1, A2_N, partial_match=(seeds_A1, seeds_A2_N), padding="naive", rng=0, n_init=5
)

# unshuffle A2_N using indices_B
P_unshuffle = make_unshuffler(sgm_naive.indices_B)
A2_N_unshuffle_seeds_naive = P_unshuffle.T @ A2_N @ P_unshuffle

A2_naive_full = np.zeros(A1.shape)
A2_naive_full[np.ix_(sgm_naive.indices_A, sgm_naive.indices_A)] = (
    A2_N_unshuffle_seeds_naive
)
# %%
A2_naive_full = np.zeros(A1.shape)
A2_naive_full[np.ix_(sgm_naive.indices_A, sgm_naive.indices_A)] = (
    A2_N_unshuffle_seeds_naive
)
# %%
A1_induced = remove_vertices(A1, nodes_to_remove)
disagreements_naive = np.linalg.norm(A1_induced - A2_N_unshuffle_seeds_naive) ** 2
print(f"Disagreements, naive padding: {int(disagreements_naive):d}")
# Disagreements, adopted padding: 9198
# %%
A1tilde = 2 * A1 - np.ones(A1.shape[0])
A2tilde_N = 2 * A2_N - np.ones(A2_N.shape[0])
A2tilde_N_padded = np.pad(A2tilde_N, [(0, nremove * n_blocks), (0, nremove * n_blocks)])
# %%
# run SGM with A1 and A2[N] with nodes removed
sgm_adopted = graph_match(
    A1, A2_N, partial_match=(seeds_A1, seeds_A2_N), padding="adopted", rng=0, n_init=5
)

# unshuffle A2[N] using the permutation identified
P_unshuffle_ad = make_unshuffler(sgm_adopted.indices_B)
A2_N_unshuffle_seeds_adopted = P_unshuffle_ad.T @ A2_N @ P_unshuffle_ad

A2_adopted_full = np.zeros(A1.shape)
A2_adopted_full[np.ix_(sgm_adopted.indices_A, sgm_adopted.indices_A)] = (
    A2_N_unshuffle_seeds_adopted
)
# %%
match_ratio_adopted = match_ratio(np.eye(A1_induced.shape[0]), P_unshuffle_ad)
print(f"Match Ratio, adopted padding: {match_ratio_adopted:.3f}")
# Match Ratio, adopted padding: 0.887

disagreements_adopted = np.linalg.norm(A1_induced - A2_N_unshuffle_seeds_adopted) ** 2
print(f"Disagreements, adopted padding: {int(disagreements_adopted):d}")
# Disagreements, adopted padding: 4186
# %%
