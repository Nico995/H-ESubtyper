import h5py
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from anndata import AnnData
from sklearn.cluster import KMeans
import pickle
from concurrent.futures import ProcessPoolExecutor

from sklearn.neighbors import BallTree
from scipy.stats import fisher_exact
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import Table2x2
from collections import defaultdict
from time import time
from sklearn import metrics

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import re


def load_features_avg(feats_root, feats_names):

    feats_paths = [
        p for p in sorted(list(feats_root.glob("*.h5"))) if p.stem in feats_names
    ]

    print(f"Found {len(feats_paths)} feature files...")

    features = []
    # coordinates = []
    samples = []
    for feat_path in tqdm(feats_paths):
        with h5py.File(feat_path, "r") as handle:
            feats = handle["features"][()]
            features.append(feats.mean(axis=0))
            samples.append(feat_path.stem)

    features = np.stack(features)

    return features, samples
    # coordinates = np.concatenate(coordinates)


def load_features(feats_root, feats_names):
    feats_paths = [
        p for p in sorted(list(feats_root.glob("*.h5"))) if p.stem in feats_names
    ]

    print(f"Found {len(feats_paths)} feature files...")

    features = []
    coordinates = []
    samples = []
    for feat_path in tqdm(feats_paths):
        with h5py.File(feat_path, "r") as handle:
            feats = handle["features"][()]
            coords = handle["coords"][()]
            features.append(feats)
            samples.append(np.array([feat_path.stem] * feats.shape[0]))
            coordinates.append(coords)

    features = np.concatenate(features, axis=0)
    samples = np.concatenate(samples, axis=0)
    coordinates = np.concatenate(coordinates, axis=0)

    return features, samples, coordinates


def load_features_averages_as_adata(feats_root, metadata, feat_id_col):
    features, samples = load_features_avg(feats_root, metadata[feat_id_col].tolist())
    adata = AnnData(features)
    adata.obs["sample"] = samples
    adata.obs = adata.obs.set_index("sample")

    adata.obs = adata.obs.merge(
        metadata.loc.set_index(feat_id_col)[:, metadata.columns],
        left_index=True,
        right_on=feat_id_col,
    ).set_index(feat_id_col)

    return adata


def load_features_as_adata(feats_root, metadata, feat_id_col):
    features, samples, coordinates = load_features(
        feats_root, metadata[feat_id_col].tolist()
    )
    adata = AnnData(features)
    adata.obs["sample"] = samples
    adata.obs = adata.obs.set_index("sample")
    adata.obsm["coordinates"] = coordinates

    adata.obs = adata.obs.merge(
        metadata.loc[:, metadata.columns],
        left_index=True,
        right_on=feat_id_col,
    )

    adata.obs.index = [
        f"{sid}-{int(x)}-{int(y)}"
        for sid, (x, y) in zip(adata.obs[feat_id_col], adata.obsm["coordinates"])
    ]

    return adata


def find_subtypes(
    data,
    n_clusters=4,
    random_state=42,
    return_centers=False,
    return_normalized_centers=False,
):
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    clusters = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_

    # Handle optional returns
    extras = []

    if return_centers:
        extras.append(centers)

    if return_normalized_centers:
        norm_centers = (centers - data.mean(axis=0)) / data.std(axis=0)
        extras.append(norm_centers)

    if not extras:
        return clusters
    else:
        return (clusters, *extras)


def save_centroids(centers, path, order=None):
    if order is not None:
        centers = centers[np.array(order)]

    with open(path, "wb") as f:
        pickle.dump(centers, f)


def fisher_1_vs_all(data, group_col, disease_col, disease_states):
    results = []

    for cluster in data[group_col].unique():
        sub = data[data[group_col] == cluster]
        contingency = pd.crosstab(sub[disease_col], columns="count")

        if disease_states[0] not in contingency.index:
            contingency.loc[disease_states[0]] = 0
        if disease_states[1] not in contingency.index:
            contingency.loc[disease_states[1]] = 0

        a = contingency.loc[disease_states[0], "count"]
        b = contingency.loc[disease_states[1], "count"]

        # Construct 2x2 table: [[pCR, RD], [¬pCR, ¬RD]]
        # In this case we assume the "¬" is all other clusters
        rest = data[data[group_col] != cluster]
        rest_tab = pd.crosstab(rest[disease_col], columns="count")
        if disease_states[0] not in rest_tab.index:
            rest_tab.loc[disease_states[0]] = 0
        if disease_states[1] not in rest_tab.index:
            rest_tab.loc[disease_states[1]] = 0

        c = rest_tab.loc[disease_states[0], "count"]
        d = rest_tab.loc[disease_states[1], "count"]

        table = np.array([[a, b], [c, d]])

        try:
            oddsratio, p = fisher_exact(table)
            ci = Table2x2(table).oddsratio_confint()
            results.append(
                {
                    "cluster": cluster,
                    "odds_ratio": oddsratio,
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                    "p_value": p,
                    "table": table.tolist(),
                }
            )
        except Exception as e:
            results.append({"cluster": cluster, "error": str(e)})

    return pd.DataFrame(results)


# LDA
# heavily borrowed from
# https://github.com/labsyspharm/scimap/blob/master/scimap/tools/spatial_lda.py
# Function
def spatial_lda(
    adata,
    x_coordinate="X_centroid",
    y_coordinate="Y_centroid",
    z_coordinate=None,
    phenotype="phenotype",
    method="radius",
    radius=30,
    knn=10,
    imageid="imageid",
    num_motifs=10,
    random_state=0,
    subset=None,
    verbose=True,
    label="spatial_lda",
    pretrained_model_path=None,
    num_workers=1,
    **kwargs,
):
    """
    Parameters:
            adata (anndata.AnnData):
                AnnData object, containing spatial gene expression data.

            x_coordinate (str, required):
                Column name in `adata` denoting the x-coordinates.

            y_coordinate (str, required):
                Column name in `adata` denoting the y-coordinates.

            z_coordinate (str, optional):
                Column name in `adata` for z-coordinates, for 3D spatial data.

            phenotype (str, required):
                Column name in `adata` indicating cell phenotype or classification.

            method (str, optional):
                Neighborhood definition method: 'radius' for fixed distance, 'knn' for K nearest neighbors.

            radius (int, optional):
                Radius defining local neighborhoods (when method='radius').

            knn (int, optional):
                Number of nearest neighbors for neighborhood definition (when method='knn').

            imageid (str, optional):
                Column name in `adata` specifying image identifiers, for analyses within specific images.

            num_motifs (int, optional):
                Number of latent motifs to identify.

            random_state (int, optional):
                Seed for random number generator, ensuring reproducibility.

            subset (str, optional):
                Specific image identifier for targeted analysis.

            verbose (bool, optional):
                If True, enables progress and informational messages.

            label (str, optional):
                Custom label for storing results in `adata.uns`.

    Returns:
            adata (anndata.AnnData):
                The input `adata` object, updated with spatial LDA results in `adata.uns[label]`.

    Example:
            ```python

            # Analyze spatial motifs using the radius method
            adata = sm.tl.spatial_lda(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                method='radius', radius=50, num_motifs=10,
                                label='lda_radius_50')

            # KNN method with specific image subset
            adata = sm.tl.spatial_lda(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                method='knn', knn=15, num_motifs=15, subset='image_01',
                                label='lda_knn_15_image_01')

            # 3D spatial data analysis using the radius method
            adata = am.tl.spatial_lda(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid', z_coordinate='Z_centroid',
                                method='radius', radius=100, num_motifs=20, label='lda_3D_radius_100')
            ```

    """

    # Function
    def spatial_lda_internal(
        adata_subset,
        x_coordinate,
        y_coordinate,
        z_coordinate,
        phenotype,
        method,
        radius,
        knn,
        imageid,
    ):

        # Print which image is being processed
        if verbose:
            print("Processing: " + str(np.unique(adata_subset.obs[imageid])))

        # Create a dataFrame with the necessary inforamtion
        if z_coordinate is not None:
            if verbose:
                print("Including Z -axis")
            data = pd.DataFrame(
                {
                    "x": adata_subset.obs[x_coordinate],
                    "y": adata_subset.obs[y_coordinate],
                    "z": adata_subset.obs[z_coordinate],
                    "phenotype": adata_subset.obs[phenotype],
                }
            )
        else:
            data = pd.DataFrame(
                {
                    "x": adata_subset.obs[x_coordinate],
                    "y": adata_subset.obs[y_coordinate],
                    "phenotype": adata_subset.obs[phenotype],
                }
            )

        # Create a DataFrame with the necessary inforamtion
        # data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'phenotype': adata_subset.obs[phenotype]})

        # Identify neighbourhoods based on the method used
        # a) KNN method

        if method == "knn":
            if verbose:
                print(
                    "Identifying the " + str(knn) + " nearest neighbours for every cell"
                )
            if z_coordinate is not None:
                tree = BallTree(data[["x", "y", "z"]], leaf_size=2)
                ind = tree.query(data[["x", "y", "z"]], k=knn, return_distance=False)
            else:
                tree = BallTree(data[["x", "y"]], leaf_size=2)
                ind = tree.query(data[["x", "y"]], k=knn, return_distance=False)
            ind = list(np.array(item) for item in ind)

        # b) Local radius method
        if method == "radius":
            if verbose:
                print(
                    "Identifying neighbours within "
                    + str(radius)
                    + " pixels of every cell"
                )
            if z_coordinate is not None:
                kdt = BallTree(data[["x", "y", "z"]], metric="euclidean")
                ind = kdt.query_radius(
                    data[["x", "y", "z"]], r=radius, return_distance=False
                )
            else:
                kdt = BallTree(data[["x", "y"]], metric="euclidean")
                ind = kdt.query_radius(
                    data[["x", "y"]], r=radius, return_distance=False
                )

        # =============================================================================
        #         if method == 'knn':
        #             if verbose:
        #                 print("Identifying the " + str(knn) + " nearest neighbours for every cell")
        #             tree = BallTree(data[['x','y']], leaf_size= 2)
        #             ind = tree.query(data[['x','y']], k=knn, return_distance= False)
        #             #ind = [np.array(x) for x in ind]
        #             ind = list(np.array(item) for item in ind)
        #
        #         # b) Local radius method
        #         if method == 'radius':
        #             if verbose:
        #                 print("Identifying neighbours within " + str(radius) + " pixels of every cell")
        #             kdt = BallTree(data[['x','y']], leaf_size= 2)
        #             ind = kdt.query_radius(data[['x','y']], r=radius, return_distance=False)
        #
        # =============================================================================

        # Map phenotype
        phenomap = dict(
            zip(list(range(len(ind))), data["phenotype"])
        )  # Used for mapping
        for i in range(len(ind)):
            ind[i] = [phenomap[letter] for letter in ind[i]]

        # return
        return ind

    # Subset a particular image if needed
    if subset is not None:
        if verbose:
            print(f"Subsetting to image: {subset}")
        adata_list = [adata[adata.obs[imageid] == subset]]
    else:
        adata_list = [
            adata[adata.obs[imageid] == i] for i in adata.obs[imageid].unique()
        ]

    # Apply function to all images
    # Create lamda function
    r_spatial_lda_internal = lambda x: spatial_lda_internal(
        adata_subset=x,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
        z_coordinate=z_coordinate,
        phenotype=phenotype,
        method=method,
        radius=radius,
        knn=knn,
        imageid=imageid,
    )
    all_data = list(map(r_spatial_lda_internal, adata_list))  # Apply function

    # combine all the data into one
    texts = np.concatenate(all_data, axis=0).tolist()

    # LDA pre-processing
    if pretrained_model_path is not None:
        if verbose:
            print(f"Loading pretrained LDA model from {pretrained_model_path}")
        lda_model = gensim.models.ldamodel.LdaModel.load(pretrained_model_path)
        id2word = lda_model.id2word
        corpus = [id2word.doc2bow(text) for text in texts]
    else:
        if verbose:
            print("Pre-Processing Spatial LDA")
        # Create Dictionary
        id2word = corpora.Dictionary(texts)
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # Build LDA model
        if verbose:
            print("Training Spatial LDA")
        try:
            if verbose:
                print("Using multicore LDA")
            lda_model = gensim.models.ldamulticore.LdaMulticore(
                corpus=corpus,
                id2word=id2word,
                num_topics=num_motifs,
                random_state=random_state,
                workers=num_workers,
                **kwargs,
            )
        except:
            if verbose:
                print("Using single core LDA")
            lda_model = gensim.models.ldamodel.LdaModel(
                corpus=corpus,
                id2word=id2word,
                num_topics=num_motifs,
                random_state=random_state,
                **kwargs,
            )

    # Compute Coherence Score
    if pretrained_model_path is None:
        if verbose:
            print("Calculating the Coherence Score")
        coherence_model_lda = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=id2word,
            coherence="c_v",
            processes=num_workers,
        )
        coherence_lda = coherence_model_lda.get_coherence()
        if verbose:
            print("\nCoherence Score: ", coherence_lda)

    # isolate the latent features
    if verbose:
        print("Gathering the latent weights")

    topic_weights = []
    for row_list in tqdm(lda_model[corpus], total=len(corpus)):
        tmp = np.zeros(num_motifs)
        for i, w in row_list:
            tmp[i] = w
        topic_weights.append(tmp)

    # conver to dataframe
    arr = pd.DataFrame(topic_weights, index=adata.obs.index).fillna(0)
    arr = arr.add_prefix("Motif_")
    adata.uns[label] = arr

    # isolate the weights of phenotypes
    if pretrained_model_path is None:

        cell_weight = pd.DataFrame(index=np.unique(adata.obs[phenotype]))
        for i in range(lda_model.num_topics):
            words_probs = dict(lda_model.show_topic(i, topn=len(id2word)))
            tmp = pd.Series(words_probs, name=f"Motif_{i}")
            cell_weight = cell_weight.join(tmp, how="outer")
        cell_weight = cell_weight.fillna(0.0)
        adata.uns[f"{label}_probability"] = cell_weight

        # save the results in anndata object
        adata.uns[label] = arr  # save the weight for each cell
        # adata.uns[str(label) + "_model"] = lda_model
        return adata, lda_model
    # return
    return adata


# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#kmeans-sparse-high-dim
def fit_and_evaluate(km, X, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    train_times = np.asarray(train_times)

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)
