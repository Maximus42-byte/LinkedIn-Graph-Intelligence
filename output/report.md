# Profile graph analysis report

## Executive summary

Data source: **synthetic fixtures**. The run processed **10 profiles**; **10** had complete coverage of the 6 schema sections. The selected combined graph has **28 edges**.
The top-three set survived **3.2%** of **1000** edge perturbations. This is a stability measure of the computed ranking, not a real-world accuracy estimate.
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
| org | 10 | 15 | 0.333 | 2 | 34.133 |
| edu | 10 | 12 | 0.267 | 2 | 4.250 |
| skill | 10 | 18 | 0.400 | 1 | 0.528 |
| semantic | 10 | 19 | 0.422 | 1 | 0.566 |
| activity | 10 | 15 | 0.333 | 1 | 3.200 |
| combined_manual | 10 | 28 | 0.622 | 1 | 0.413 |
| combined_equal | 10 | 24 | 0.533 | 1 | 0.406 |
Semantic vectors used **TF-IDF + TruncatedSVD (LSA)**. Skill edges retain both Jaccard and TF-IDF cosine; semantic edges retain embedding and TF-IDF cosine alternatives. The chosen skill metric is **jaccard** and semantic metric is **lsa**.
The org layer ignores employers present in more than **80%** of profiles; ignored: **FableForge Analytics (Fictional) (10/10 profiles)**. Shared past employers still contribute when dates overlap.
The skill and semantic layers keep each profile's **3** highest-scoring neighbours with positive weight at least **0.050** and **0.050**, respectively. An edge survives if either endpoint selects the other; equal scores are broken by profile ID.

## Key people and roles

| Rank | ID | Person | Roles | Borda | Weighted degree | Betweenness | PageRank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | p01 | Synthetic Persona 01 | hub, influencer | 88 | 2.627 | 0.000 | 0.109 |
| 2 | p10 | Synthetic Persona 10 | hub, bridge, influencer | 77 | 2.523 | 0.556 | 0.105 |
| 3 | p02 | Synthetic Persona 02 | hub | 66 | 2.497 | 0.000 | 0.103 |
| 4 | p05 | Synthetic Persona 05 | bridge, influencer | 64 | 2.480 | 0.111 | 0.104 |
| 5 | p06 | Synthetic Persona 06 | peripheral | 52 | 2.467 | 0.000 | 0.104 |
Borda and mean reciprocal rank aggregate all centrality methods. Roles use configurable quantiles; weighted degree breaks hub ties in a dense graph.

## Method comparison

| Method A | Method B | Kendall tau | Spearman rho |
| --- | --- | --- | --- |
| degree | weighted_degree | 0.581 | 0.706 |
| betweenness | pagerank | 0.083 | 0.069 |
| pagerank | eigenvector | 0.689 | 0.770 |
| constraint | effective_size | -0.600 | -0.648 |
| closeness | harmonic | 0.511 | 0.648 |
Manual-weight and equal-normalised graph rankings have Kendall tau **0.689**. The two graph exports are retained in `graph.json`.

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
| louvain | 3 | 0 | 0.461 | — | — |
| greedy_modularity | 3 | 0 | 0.461 | — | — |
| label_propagation | 3 | 0 | 0.461 | — | — |
| girvan_newman | 4 | 0 | 0.143 | — | — |
| spectral | 3 | 0 | 0.461 | — | — |
| tfidf_kmeans | 3 | 0 | — | 0.283 | 1.104 |
| ward | 3 | 0 | — | 0.283 | 1.104 |
| embedding_kmeans | 3 | 0 | — | 0.353 | 0.983 |
| hdbscan | 3 | 0 | — | 0.353 | 0.983 |
| nmf_topics | 3 | 0 | — | 0.280 | 1.102 |
All pairwise clustering ARI/NMI scores are in `tables/clustering_pairwise.csv`. The skill projection yielded **3** skill families. Feature embeddings used **TF-IDF + TruncatedSVD (LSA)**.

## Validation and stability

| Check | Draws | Top-three retention | Kendall tau (mean/median) | Cluster ARI median |
| --- | --- | --- | --- | --- |
| Edge perturbation | 1000 | 0.032 | 0.209 | — |
| Weight sensitivity | 100 | 0.410 | 0.750 | 1.000 |
Edge perturbation removed **3** edges and added **3** per draw. Its mean Kendall tau was **0.209** with a central interval of **-0.22–0.66**. Average ranks preserve score ties, and the top-three set includes all profiles tied at the cutoff.
| ID | Baseline rank | Rank low | Rank high | Top-three frequency |
| --- | --- | --- | --- | --- |
| p01 | 2.000 | 1.000 | 10.000 | 0.429 |
| p02 | 4.000 | 1.000 | 10.000 | 0.413 |
| p03 | 7.000 | 1.000 | 10.000 | 0.283 |
| p04 | 8.500 | 1.000 | 10.000 | 0.275 |
| p05 | 3.000 | 1.000 | 10.000 | 0.409 |
| p06 | 5.000 | 1.000 | 10.000 | 0.219 |
| p07 | 6.000 | 1.000 | 10.000 | 0.232 |
| p08 | 8.500 | 1.000 | 10.000 | 0.255 |
| p09 | 10.000 | 1.000 | 10.000 | 0.166 |
| p10 | 1.000 | 1.000 | 9.000 | 0.391 |
| Null model | Statistic | Samples | Observed | Null mean | z | Empirical upper p |
| --- | --- | --- | --- | --- | --- | --- |
| degree_preserving | modularity | 100 | 0.461 | 0.223 | 5.215 | 0.010 |
| degree_preserving | top_weighted_degree | 100 | 2.627 | 3.984 | -2.377 | 1.000 |
| degree_preserving | top_betweenness | 100 | 0.556 | 0.468 | 0.765 | 0.277 |
| degree_preserving | top_pagerank | 100 | 0.109 | 0.162 | -2.433 | 1.000 |
| gnm | modularity | 100 | 0.461 | 0.222 | 6.028 | 0.010 |
| gnm | top_weighted_degree | 100 | 2.627 | 3.954 | -2.393 | 1.000 |
| gnm | top_betweenness | 100 | 0.556 | 0.448 | 0.937 | 0.248 |
| gnm | top_pagerank | 100 | 0.109 | 0.161 | -2.434 | 1.000 |
n=10 is too small for a statistically informative null comparison; empirical p-values and z-scores are descriptive only.
| Dropped layer | Edges left | Rank Kendall tau | Cluster ARI | Top-three retained |
| --- | --- | --- | --- | --- |
| org | 28 | 0.854 | 1.000 | yes |
| edu | 27 | 0.899 | 0.659 | yes |
| skill | 25 | 0.584 | 1.000 | no |
| semantic | 21 | 0.477 | 1.000 | no |
| activity | 26 | 0.719 | 1.000 | yes |
Dirichlet weight draws gave a rank Kendall tau interval of **0.36–0.98** and a cluster ARI interval of **0.66–1.00**.
**Synthetic validation:** comparison with labels used to construct the fixtures. It measures recovery of built-in structure, not real-world accuracy.
| Method | ARI | NMI |
| --- | --- | --- |
| embedding_kmeans | 0.815 | 0.906 |
| girvan_newman | 0.167 | 0.542 |
| greedy_modularity | 0.815 | 0.906 |
| hdbscan | 0.815 | 0.906 |
| label_propagation | 0.815 | 0.906 |
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
Organisational and engagement degree have Spearman rho **0.195**; their edge sets have Jaccard **0.765**. Topic and lexical consistency are both shown because their judgments can differ.
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
