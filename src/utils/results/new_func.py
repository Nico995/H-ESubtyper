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
    load_model=False,
    lda_model_path=None,
    **kwargs,
):
    # Internal function
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
        if verbose:
            print("Processing: " + str(np.unique(adata_subset.obs[imageid])))

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

        # Neighborhoods
        if method == "knn":
            if verbose:
                print(f"Identifying the {knn} nearest neighbours for every cell")
            coords = data[["x", "y", "z"]] if z_coordinate else data[["x", "y"]]
            tree = BallTree(coords, leaf_size=2)
            ind = tree.query(coords, k=knn, return_distance=False)
        elif method == "radius":
            if verbose:
                print(f"Identifying neighbours within {radius} pixels of every cell")
            coords = data[["x", "y", "z"]] if z_coordinate else data[["x", "y"]]
            tree = BallTree(coords, metric="euclidean")
            ind = tree.query_radius(coords, r=radius, return_distance=False)

        # Map indices to phenotypes
        phenomap = dict(zip(range(len(ind)), data["phenotype"]))
        for i in range(len(ind)):
            ind[i] = [phenomap[pid] for pid in ind[i]]

        return ind

    # Subset image(s)
    if subset is not None:
        adata_list = [adata[adata.obs[imageid] == subset]]
    else:
        adata_list = [
            adata[adata.obs[imageid] == i] for i in adata.obs[imageid].unique()
        ]

    # Extract phenotype neighborhoods
    all_data = list(
        map(
            lambda x: spatial_lda_internal(
                x,
                x_coordinate,
                y_coordinate,
                z_coordinate,
                phenotype,
                method,
                radius,
                knn,
                imageid,
            ),
            adata_list,
        )
    )
    texts = np.concatenate(all_data, axis=0).tolist()

    # Corpus prep
    if load_model:
        if verbose:
            print(f"Loading pretrained LDA model from {lda_model_path}")
        lda_model = gensim.models.ldamodel.LdaModel.load(lda_model_path)
        id2word = lda_model.id2word
        corpus = [id2word.doc2bow(text) for text in texts]
    else:
        if verbose:
            print("Pre-Processing Spatial LDA")
        id2word = corpora.Dictionary(texts)
        corpus = [id2word.doc2bow(text) for text in texts]
        if verbose:
            print("Training Spatial LDA")
        try:
            lda_model = gensim.models.ldamulticore.LdaMulticore(
                corpus=corpus,
                id2word=id2word,
                num_topics=num_motifs,
                random_state=random_state,
                **kwargs,
            )
        except:
            lda_model = gensim.models.ldamodel.LdaModel(
                corpus=corpus,
                id2word=id2word,
                num_topics=num_motifs,
                random_state=random_state,
                **kwargs,
            )
        if lda_model_path:
            lda_model.save(lda_model_path)
            if verbose:
                print(f"LDA model saved to: {lda_model_path}")

    # Coherence (only when training)
    if not load_model and verbose:
        print("Calculating the Coherence Score")
        coherence_model_lda = CoherenceModel(
            model=lda_model, texts=texts, dictionary=id2word, coherence="c_v"
        )
        coherence_lda = coherence_model_lda.get_coherence()
        print("\nCoherence Score:", coherence_lda)

    # Inference: get topic weights per neighborhood
    if verbose:
        print("Gathering the latent weights")
    topic_weights = []
    for row_list in lda_model[corpus]:
        tmp = np.zeros(num_motifs)
        for i, w in row_list:
            tmp[i] = w
        topic_weights.append(tmp)

    arr = pd.DataFrame(topic_weights, index=adata.obs.index).fillna(0)
    arr = arr.add_prefix("Motif_")
    adata.uns[label] = arr

    # Topic-word matrix (only if training)
    if not load_model:
        pattern = r'(\d\.\d+)."(.*?)"'
        cell_weight = pd.DataFrame(index=np.unique(adata.obs[phenotype]))
        for i, topic_str in lda_model.print_topics():
            tmp = pd.DataFrame(re.findall(pattern, topic_str))
            tmp.index = tmp[1]
            tmp = tmp.drop(columns=1)
            tmp.columns = [f"Motif_{i}"]
            cell_weight = cell_weight.merge(
                tmp, how="outer", left_index=True, right_index=True
            )
        adata.uns[f"{label}_probability"] = cell_weight.fillna(0).astype(float)

    # Always store the model
    adata.uns[f"{label}_model"] = lda_model

    return adata
