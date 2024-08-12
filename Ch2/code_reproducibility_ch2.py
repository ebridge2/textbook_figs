#%%  section 2.2

mode = "png"

import matplotlib

font = {"family": "Dejavu Sans", "weight": "bold", "size": 20}

matplotlib.rc("font", **font)
# %%
import os
import urllib
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from graspologic.utils import import_edgelist
import numpy as np
import glob
from tqdm import tqdm

np.random.seed(0)


# the AWS bucket the data is stored in
BUCKET_ROOT = "open-neurodata"
parcellation = "Schaefer400"
FMRI_PREFIX = (
    "m2g/Functional/BNU1-11-12-20-m2g-func/Connectomes/"
    + parcellation
    + "_space-MNI152NLin6_res-2x2x2.nii.gz/"
)
FMRI_PATH = os.path.join("datasets", "fmri")  # the output folder
DS_KEY = "abs_edgelist"  # correlation matrices for the networks to exclude


def fetch_fmri_data(
    bucket=BUCKET_ROOT, fmri_prefix=FMRI_PREFIX, output=FMRI_PATH, name=DS_KEY
):
    """
    A function to fetch fMRI connectomes from AWS S3.
    """
    # check that output directory exists
    if not os.path.isdir(FMRI_PATH):
        os.makedirs(FMRI_PATH)
    # start boto3 session anonymously
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    # obtain the filenames
    bucket_conts = s3.list_objects(Bucket=bucket, Prefix=fmri_prefix)["Contents"]
    for s3_key in tqdm(bucket_conts):
        # get the filename
        s3_object = s3_key["Key"]
        # verify that we are grabbing the right file
        if name not in s3_object:
            op_fname = os.path.join(FMRI_PATH, str(s3_object.split("/")[-1]))
            if not os.path.exists(op_fname):
                s3.download_file(bucket, s3_object, op_fname)


def read_fmri_data(path=FMRI_PATH):
    """
    A function which loads the connectomes as adjacency matrices.
    """
    # import edgelists with graspologic
    # edgelists will be all of the files that end in a csv
    networks = [
        import_edgelist(fname) for fname in tqdm(glob.glob(os.path.join(path, "*.csv")))
    ]
    return np.stack(networks, axis=0)
# %%
fetch_fmri_data()
As = read_fmri_data()
# %%
from graphbook_code import heatmap

A = As[0]
ax = heatmap(A, vmin=0, vmax=1, title="Heatmap of Functional Connectome")
# %%
from graphbook_code import heatmap
from matplotlib import pyplot as plt


A = As[0]

fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 2]}, figsize=(20, 6))
heatmap(
    A,
    vmin=0,
    vmax=1,
    ax=axs[0],
    xticks=[0, 199, 399],
    xticklabels=[1, 200, 400],
    yticks=[0, 199, 399],
    yticklabels=[1, 200, 400],
    ytitle="Brain Area",
    xtitle="Brain Area",
    title="(A) Heatmap of Functional Connectome",
)
import seaborn as sns

sns.histplot(A.flatten(), ax=axs[1], bins=50, color="gray")
axs[1].set_xlabel("Edge weight")
axs[1].set_title("(B) Histogram of functional connectome edge-weights", size=20)
# fig.savefig("Figures/raw.svg")
os.makedirs("Figures", exist_ok=True)
fig.savefig("Figures/raw.{}".format(mode))

# %% section 2.3
def remove_isolates(A):
    """
    A function which removes isolated nodes from the
    adjacency matrix A.
    """
    degree = A.sum(axis=0)  # sum along the rows to obtain the node degree
    out_degree = A.sum(axis=1)
    A_purged = A[~(degree == 0), :]
    A_purged = A_purged[:, ~(degree == 0)]
    print("Purging {:d} nodes...".format((degree == 0).sum()))
    return A_purged


A = remove_isolates(A)
# Purging 0 nodes...
# %%
import matplotlib.pyplot as plt
from graphbook_code import heatmap

A_abs = np.abs(A)
fig, axs = plt.subplots(1, 3, figsize=(21, 6))
heatmap(A, ax=axs[0], title="Human Connectome, Raw", vmin=np.min(A), vmax=1)
heatmap(A_abs, ax=axs[1], title="Human Connectome, Absolute", vmin=np.min(A), vmax=1)
heatmap(A_abs - A, ax=axs[2], title="Difference(Absolute - Raw)", vmin=0, vmax=1)
# %%
from sklearn.base import TransformerMixin, BaseEstimator


class CleanData(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        print("Cleaning data...")
        Acleaned = remove_isolates(X)
        A_abs_cl = np.abs(Acleaned)
        self.A_ = A_abs_cl
        return self.A_


data_cleaner = CleanData()
A_clean = data_cleaner.transform(A)
# Cleaning data...
# Purging 0 nodes...
# %%
from graspologic.utils import binarize

threshold = 0.4
A_bin = binarize(A_clean > threshold)
# %%
from graspologic.utils import pass_to_ranks

A_ptr = pass_to_ranks(A_clean)
# %%
import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
sns.histplot(A_clean[A_clean > 0].flatten(), ax=axs[0], color="gray")
axs[0].set_xlabel("Edge weight")
axs[0].set_title("Histogram of human connectome, non-zero edge weights")
sns.histplot(A_ptr[A_ptr > 0].flatten(), ax=axs[1], color="gray")
axs[1].set_xlabel("ptr(Edge weight)")
axs[1].set_title("Histogram of human connectome, passed-to-ranks")

plt.tight_layout()
# %%
class FeatureScaler(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        print("Scaling edge-weights...")
        A_scaled = pass_to_ranks(X)
        return A_scaled


feature_scaler = FeatureScaler()
A_cleaned_scaled = feature_scaler.transform(A_clean)
# Scaling edge-weights...
# %%
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline(
    [
        ("cleaner", CleanData()),
        ("scaler", FeatureScaler()),
    ]
)

A_xfm = num_pipeline.fit_transform(A)
# Cleaning data...
# Purging 0 nodes...
# Scaling edge-weights..
# %%
A_xfm2 = num_pipeline.fit_transform(As[1])
# Cleaning data...
# Purging 0 nodes...
# Scaling edge-weights...
# %%  section 2.4

from graspologic.embed import AdjacencySpectralEmbed

embedding = AdjacencySpectralEmbed(n_components=3, svd_seed=0).fit_transform(A_xfm)
# %%
from graspologic.plot import pairplot

_ = pairplot(embedding, title="Spectral Embedding for connectome")
# %%
from sklearn.cluster import KMeans

labels = KMeans(n_clusters=2, random_state=0).fit_predict(embedding)
_ = pairplot(
    embedding,
    labels=labels,
    legend_name="Predicter Clusters",
    title="KMeans clustering",
)
# %%
from graspologic.cluster import KMeansCluster

labels = KMeansCluster(max_clusters=10, random_state=0).fit_predict(embedding)
_ = pairplot(
    embedding,
    labels=labels,
    title="KMeans clustering, automatic selection",
    legend_name="Predicted Clusters",
)
# %%
from graspologic.cluster import AutoGMMCluster

labels = AutoGMMCluster(max_components=10, random_state=0).fit_predict(embedding)
_ = pairplot(
    embedding,
    labels=labels,
    title="AutoGMM Clustering, automatic selection",
    legend_name="Predicted Clusters",
)
# %%  section 2.5

from graspologic.embed import MultipleASE

# transform all the networks with pipeline utility
As_xfm = [num_pipeline.fit_transform(A) for A in As]
# and embed them
embedding = MultipleASE(n_components=5, svd_seed=0).fit_transform(As_xfm)
_ = pairplot(embedding, title="Multiple spectral embedding of all connectomes")

# %%
labels = AutoGMMCluster(max_components=10).fit_predict(embedding)
_ = pairplot(
    embedding,
    labels=labels,
    title="Multiple spectral embedding of all connectomes",
    legend_name="Predicted Clusters",
)
# %%
from urllib import request
import json
import pandas as pd
from pathlib import Path

coord_dest = os.path.join(FMRI_PATH, "coordinates.json")
with open(coord_dest) as coord_f:
    coords = []
    for roiname, contents in json.load(coord_f)["rois"].items():
        try:
            if roiname != "0":
                coord_roi = {
                    "x": contents["center"][0],
                    "y": contents["center"][1],
                    "z": contents["center"][2],
                }
                coords.append(coord_roi)
        except:
            continue

coords_df = pd.DataFrame(coords)
# %%
import matplotlib.image as mpimg

coords_df["Community"] = labels
coords_df["Community"] = coords_df["Community"].astype("category")
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
axs[0].imshow(mpimg.imread("./Images/lobes.png"))
axs[0].set_axis_off()
sns.scatterplot(x="y", y="z", data=coords_df, hue="Community", ax=axs[1])
# %%
import datasets.dice as dice

# obtain the Yeo7 parcellation
group_dest = os.path.join("./datasets/", "Yeo-7_space-MNI152NLin6_res-2x2x2.nii.gz")
request.urlretrieve(
    "https://github.com/neurodata/neuroparc/"
    + "blob/master/atlases/label/Human/"
    + "Yeo-7_space-MNI152NLin6_res-2x2x2.nii.gz?raw=true",
    group_dest,
)
# obtain the Shaefer parcellation
roi_dest = os.path.join(
    "./datasets/", parcellation + "_space-MNI152NLin6_res-2x2x2.nii.gz"
)
request.urlretrieve(
    "https://github.com/neurodata/neuroparc/"
    + "blob/master/atlases/label/Human/"
    + parcellation
    + "_space-MNI152NLin6_res-2x2x2.nii.gz?raw=true",
    roi_dest,
)
# decipher which Schaefer labels fall within Yeo7 regions
dicemap, _, _ = dice.dice_roi(
    "./datasets/",
    "./datasets",
    "Yeo-7_space-MNI152NLin6_res-2x2x2.nii.gz",
    parcellation + "_space-MNI152NLin6_res-2x2x2.nii.gz",
    verbose=False,
)
actual_cluster = np.argmax(dicemap, axis=0)[1:] - 1
# %%
import contextlib
from sklearn.metrics import confusion_matrix
from graphbook_code import cmaps

# make confusion matrix
cf_matrix = confusion_matrix(actual_cluster, labels)

# and plot it
ax = sns.heatmap(cf_matrix, cmap=cmaps["sequential"])
ax.set_title("Confusion matrix")
ax.set_ylabel("True Parcel")
ax.set_xlabel("Predicted Community")
# %%
