# Profile graph analysis report

## Executive summary

Data source: **synthetic fixtures**. The run processed **10 profiles**; **10** had complete coverage of the 6 schema sections. The selected combined graph has **45 edges**.
The top-three set survived **9.8%** of **1000** edge perturbations. This is a stability measure of the computed ranking, not a real-world accuracy estimate.
With n=10, these results are illustrative and not statistically meaningful.

## Data and coverage

All fixture pages carry a visible SYNTHETIC banner and a synthetic meta tag.
| Section | Covered | Profiles | Fraction |
| --- | --- | --- | --- |
| basic | 10 | 10 | 1.000 |
| experience | 10 | 10 | 1.000 |
| education | 10 | 10 | 1.000 |
| skills | 10 | 10 | 1.000 |
| interests | 10 | 10 | 1.000 |
| activity | 10 | 10 | 1.000 |

## Graph statistics

| Layer | Nodes | Edges | Density | Components | Mean edge weight |
| --- | --- | --- | --- | --- | --- |
| org | 10 | 45 | 1.000 | 1 | 52.911 |
| edu | 10 | 12 | 0.267 | 2 | 4.250 |
| skill | 10 | 45 | 1.000 | 1 | 0.329 |
| semantic | 10 | 45 | 1.000 | 1 | 0.374 |
| activity | 10 | 15 | 0.333 | 1 | 3.200 |
| combined_manual | 10 | 45 | 1.000 | 1 | 0.433 |
| combined_equal | 10 | 45 | 1.000 | 1 | 0.278 |
Semantic vectors used **TF-IDF + TruncatedSVD (LSA)**. Skill edges retain both Jaccard and TF-IDF cosine; semantic edges retain embedding and TF-IDF cosine alternatives. The chosen skill metric is **jaccard** and semantic metric is **lsa**.

## Key people and roles

| Rank | ID | Person | Roles | Borda | Weighted degree | Betweenness | PageRank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | p03 | Synthetic Persona 03 | hub, bridge, influencer | 85 | 4.150 | 0.028 | 0.105 |
| 2 | p06 | Synthetic Persona 06 | hub, bridge, influencer | 74 | 4.092 | 0.028 | 0.104 |
| 3 | p01 | Synthetic Persona 01 | hub, influencer | 70 | 4.074 | 0.000 | 0.104 |
| 4 | p02 | Synthetic Persona 02 | peripheral | 55 | 3.995 | 0.000 | 0.102 |
| 5 | p04 | Synthetic Persona 04 | peripheral | 51 | 3.950 | 0.000 | 0.101 |
Borda and mean reciprocal rank aggregate all centrality methods. Roles use configurable quantiles; weighted degree breaks hub ties in a dense graph.

## Method comparison

| Method A | Method B | Kendall tau | Spearman rho |
| --- | --- | --- | --- |
| degree | weighted_degree | — | — |
| betweenness | pagerank | 0.201 | 0.208 |
| pagerank | eigenvector | 0.956 | 0.988 |
| constraint | effective_size | -0.156 | -0.139 |
| closeness | harmonic | 0.644 | 0.782 |
Manual-weight and equal-normalised graph rankings have Kendall tau **0.822**. The two graph exports are retained in `graph.json`.

## Clustering results

Feature k selection chose **3** groups; silhouette choice **3**, elbow choice **3**.
| k | Silhouette | Inertia |
| --- | --- | --- |
| 2 | 0.215 | 5.058 |
| 3 | 0.283 | 3.281 |
| 4 | 0.249 | 2.483 |
| 5 | 0.215 | 1.908 |
| Method | Clusters | Noise | Modularity | Silhouette | Davies–Bouldin |
| --- | --- | --- | --- | --- | --- |
| louvain | 3 | 0 | 0.170 | — | — |
| greedy_modularity | 3 | 0 | 0.170 | — | — |
| label_propagation | 1 | 0 | 0.000 | — | — |
| girvan_newman | 4 | 0 | 0.015 | — | — |
| spectral | 3 | 0 | 0.170 | — | — |
| tfidf_kmeans | 3 | 0 | — | 0.283 | 1.104 |
| ward | 3 | 0 | — | 0.283 | 1.104 |
| embedding_kmeans | 3 | 0 | — | 0.353 | 0.983 |
| hdbscan | 3 | 0 | — | 0.353 | 0.983 |
| nmf_topics | 3 | 0 | — | 0.280 | 1.102 |
All pairwise clustering ARI/NMI scores are in `tables/clustering_pairwise.csv`. The skill projection yielded **3** skill families. Feature embeddings used **TF-IDF + TruncatedSVD (LSA)**.

## Validation and stability

| Check | Draws | Top-three retention | Kendall tau (mean/median) | Cluster ARI median |
| --- | --- | --- | --- | --- |
| Edge perturbation | 1000 | 0.098 | 0.409 | — |
| Weight sensitivity | 100 | 0.390 | 0.733 | 1.000 |
Edge perturbation removed **4** edges and added **0** per draw. Its mean Kendall tau was **0.409** with a central interval of **-0.02–0.78**.
| ID | Baseline rank | Rank low | Rank high | Top-three frequency |
| --- | --- | --- | --- | --- |
| p01 | 3 | 1.000 | 10.000 | 0.499 |
| p02 | 4 | 1.000 | 10.000 | 0.356 |
| p03 | 1 | 1.000 | 9.000 | 0.612 |
| p04 | 6 | 2.000 | 10.000 | 0.268 |
| p05 | 7 | 2.000 | 10.000 | 0.178 |
| p06 | 2 | 1.000 | 10.000 | 0.550 |
| p07 | 8 | 3.000 | 10.000 | 0.096 |
| p08 | 10 | 5.000 | 10.000 | 0.000 |
| p09 | 9 | 3.000 | 10.000 | 0.029 |
| p10 | 5 | 1.000 | 10.000 | 0.412 |
| Null model | Statistic | Samples | Observed | Null mean | z | Empirical upper p |
| --- | --- | --- | --- | --- | --- | --- |
| degree_preserving | modularity | 100 | 0.372 | 0.187 | 5.358 | 0.010 |
| degree_preserving | top_weighted_degree | 100 | 3.425 | 4.560 | -2.599 | 0.990 |
| degree_preserving | top_betweenness | 100 | 0.222 | 0.338 | -1.205 | 0.970 |
| degree_preserving | top_pagerank | 100 | 0.119 | 0.157 | -2.793 | 0.990 |
| gnm | modularity | 100 | 0.372 | 0.185 | 4.870 | 0.010 |
| gnm | top_weighted_degree | 100 | 3.425 | 4.413 | -1.644 | 0.990 |
| gnm | top_betweenness | 100 | 0.222 | 0.326 | -1.076 | 0.931 |
| gnm | top_pagerank | 100 | 0.119 | 0.153 | -1.751 | 0.990 |
n=10 is too small for a statistically informative null comparison; empirical p-values and z-scores are descriptive only.
The combined graph was too dense for useful edge swaps, so the null comparison used a backbone containing the **22 strongest edges**; this changes the tested graph.
| Dropped layer | Edges left | Rank Kendall tau | Cluster ARI | Top-three retained |
| --- | --- | --- | --- | --- |
| org | 45 | 0.378 | 1.000 | no |
| edu | 45 | 0.778 | 1.000 | yes |
| skill | 45 | 0.600 | 1.000 | yes |
| semantic | 45 | 0.600 | 1.000 | no |
| activity | 45 | 0.867 | 1.000 | yes |
Dirichlet weight draws gave a rank Kendall tau interval of **0.40–0.89** and a cluster ARI interval of **1.00–1.00**.
**Synthetic validation:** comparison with labels used to construct the fixtures. It measures recovery of built-in structure, not real-world accuracy.
| Method | ARI | NMI |
| --- | --- | --- |
| embedding_kmeans | 0.815 | 0.906 |
| girvan_newman | 0.167 | 0.542 |
| greedy_modularity | 0.815 | 0.906 |
| hdbscan | 0.815 | 0.906 |
| label_propagation | 0.000 | 0.000 |
| louvain | 0.815 | 0.906 |
| nmf_topics | 0.815 | 0.906 |
| spectral | 0.815 | 0.906 |
| tfidf_kmeans | 0.815 | 0.906 |
| ward | 0.815 | 0.906 |

## Career, education, activity, and similarity findings

Dated job history yielded **4** company-to-company routes at observation month **2025-04**. Exactly **1** recorded skill has one holder.
| Talent source | Outbound people | Transitions |
| --- | --- | --- |
| Cloudcastle Systems (Fictional) | 4 | 4 |
| Moonbeam Modelworks (Fictional) | 4 | 4 |
| Cipherkeep Works (Fictional) | 3 | 3 |
| Single-holder skill | Holder |
| --- | --- |
| synthetic protocol archaeology | p09 |
| ID | Posts | Reactions | Engagement degree | Topic consistency | Lexical consistency |
| --- | --- | --- | --- | --- | --- |
| p01 | 2 | 18 | 9.000 | 0.988 | 0.445 |
| p02 | 2 | 16 | 8.000 | 0.993 | 0.462 |
| p03 | 2 | 19 | 11.000 | 0.992 | 0.519 |
| p04 | 2 | 12 | 11.000 | 0.943 | 0.615 |
| p05 | 2 | 22 | 8.000 | 0.882 | 0.357 |
| p06 | 2 | 18 | 9.000 | 0.228 | 0.353 |
| p07 | 2 | 13 | 11.000 | 0.995 | 0.796 |
| p08 | 2 | 12 | 8.000 | 0.951 | 0.383 |
| p09 | 2 | 11 | 9.000 | 0.892 | 0.000 |
| p10 | 2 | 14 | 12.000 | 0.695 | 0.421 |
Organisational and engagement degree have Spearman rho **0.798**; their edge sets have Jaccard **0.333**. Topic and lexical consistency are both shown because their judgments can differ.
| Post topic | Top terms |
| --- | --- |
| 0 | synthetic, checks, pipeline, imaginary, engineering |
| 1 | security, fictional, sandbox, team, distributed |
| 2 | cloud, infrastructure, clear, combined, container |
The positive-similarity nearest-colleague graph used k=**3** and contains **19 edges**. Its cosine and Jaccard matrices are visualised in `similarity_heatmap.png`.
| ID | Nearest colleagues | Cosine values |
| --- | --- | --- |
| p01 | p02, p03, p10 | 0.80658681, 0.76572491, 0.36166768 |
| p02 | p01, p03, p05 | 0.80658681, 0.77409512, 0.44040808 |
| p03 | p02, p01, p06 | 0.77409512, 0.76572491, 0.33125854 |
| p04 | p06, p05, p10 | 0.72300196, 0.66426606, 0.29641125 |
| p05 | p06, p04, p02 | 0.85625652, 0.66426606, 0.44040808 |
| p06 | p05, p04, p02 | 0.85625652, 0.72300196, 0.34514107 |
| p07 | p09, p08, p06 | 0.80392994, 0.80218407, 0.16687912 |
| p08 | p09, p07, p05 | 0.89482347, 0.80218407, 0.26698566 |
| p09 | p08, p07, p06 | 0.89482347, 0.80392994, 0.18553849 |
| p10 | p01, p05, p06 | 0.36166768, 0.34963241, 0.33965979 |

## Limitations

The graph contains **10** observed profiles. At this size, method rankings, modularity, p-values, and role labels can change sharply after a few edges change. The perturbation and weight-sensitivity tables quantify that dependence.
All people, organisations, schools, jobs, skills, and posts are synthetic. Ground-truth agreement reflects designed fixture structure and does not validate real-person predictions.
An edge means observed overlap or interaction under the configured rule; it does not prove a real relationship. Export and live sources can leave fields uncovered, and missing skills or activity must not be treated as absence in reality.
Optional sentence-transformers vectors require an already available local model. The default run uses the offline fallback.
