# Assumptions and design notes

This repository treats the brief as an extraction and analysis demonstration. The default source is visibly synthetic HTML because no real-person profile collection was authorised and a public page may be blocked by robots rules, login, or the site's terms. The offline fixture run processed the generated profiles through the same HTML parser used by the opt-in live adapter. Missing fields remain empty with false coverage; no score treats an absent field as a proven absence of a skill or relationship.

## A. If extraction ran online: HTTP handling

[`live.py`](src/lgi/sources/live.py) is the implementation, but no live request was made during development. It uses one `requests.Session` with a project-identifying User-Agent, `Accept` and `Accept-Language`, an explicit timeout, disabled ambient proxies/credentials, and cleared cookies. It checks cached `robots.txt` rules before each profile URL, rate limits by host with delay and jitter, and retries only HTTP 429/5xx or network errors using exponential backoff and `Retry-After`. It refuses redirects and stops a URL on HTTP 999, a login redirect, a login form, or robots disallow; the adapter yields a false-coverage record with the reason. A fixed, authorised corporate egress proxy could be appropriate for ordinary network policy, but rotating proxies or proxies used to bypass access controls would not be. Public visibility does not remove legal, privacy, or terms-of-service obligations; the operator must assess them before opt-in use.

## B. Processing with a queue (Kafka/Redis)

A producer would enqueue profile URLs. Workers would fetch, parse, and write by a SHA-256 URL idempotency key. At-least-once delivery can replay a job, so parsing and writes must be idempotent. Retryable failures need backoff; exhausted jobs go to a dead-letter queue. Deduplication uses the same key, and partitioning by host would let workers enforce a shared host rate limit. [`ingest_demo.py`](src/lgi/ingest_demo.py) simulates this with a stdlib queue, retries, deduplication, and dead letters. Kafka and Redis are not run here.

## C. Storage strategy at scale

Immutable raw HTML would live in object storage such as S3/MinIO so a schema or parser change can trigger re-parsing. Normalised, schema-versioned profiles would live in PostgreSQL or MongoDB. Neo4j could serve interactive graph traversal; an edge list in Parquet is simpler for batch analysis. Parquet/lakehouse tables would support analytical scans. Real-person retention would need minimisation, access control, deletion requests, and removal of raw as well as derived records. This repository stores only synthetic default data.

## D. Analysis design and assumptions

Separate organisation, education, skill, semantic, and observed-engagement layers prevent one type of evidence from being mistaken for another. The org layer excludes employers appearing in more than `org_ignore_employer_share` of profiles; in the fixture run it excludes the fictional focal employer while retaining shared past employers. Skill and semantic layers use symmetrised top-`similarity_knn_k` neighbours plus their configured positive minimum weights, with profile ID breaking equal-score ties. These rules were chosen without reading ground truth or tuning ARI. Both individual layers and fixed-weight versus equal-normalised combinations are retained.

The configured `reference_month` is `2026-01` in the tested run. It is the inclusive end month for open-ended jobs in employer overlap and career tenure; it is not derived from the most recent activity date. The same config value now drives both modules.

Multiple centralities represent distinct notions of importance, while graph and feature clusterings expose method disagreement. The key-people ranking combines all centrality methods; stability uses weighted degree, weighted betweenness, and PageRank. The tested run's weighted-degree coefficient of variation is 0.105, and the absolute weighted-degree gap between key-people ranks 1 and 5 is 0.161. Near-ties are consistent with unstable ranks, without establishing causation. Constraint and effective size are expected to correlate negatively, so their observed Kendall tau of -0.600 is not a disagreement.

Perturbation, degree-preserving and G(n,m) nulls, ablation, and weight draws test dependence on edges and weights. Exact top-three retention remains the main stability measure: 3.2% in the tested run. After seeing that low value, mean top-three overlap (1.229 versus nominal chance 0.900 for sets of three among ten) and top-one retention (10.4%) were added as context, without changing the primary metric or analysis parameters. Validation uses average ranks and includes cutoff ties in top-k sets; the nominal chance calculation assumes exactly three members. Per-person intervals now show the 5th and 95th percentiles alongside true minimum and maximum ranks. The small node count makes null-model p-values descriptive only; a strongest-edge backbone is used if the combined graph is too dense for swaps.

The fixture truth has four labels because Bridge is separate, while nine methods returned three clusters. Merging Bridge into each team gives a three-cluster ceiling of ARI 0.815 and NMI 0.906; nine methods reached it, recovering all three teams on these clean synthetic fixtures. Girvan–Newman was below it. Nine methods produced identical partitions; this is synthetic structure recovery, not evidence of real-world accuracy. HDBSCAN uses sklearn's actual estimator output, with no substituted fallback, and matched embedding KMeans on the tested run. Girvan–Newman uses unweighted edge betweenness to generate cuts, then scores them with weighted modularity on the same combined graph as the other graph methods.

The fixture generator marked p09 as the designed post/skill mismatch, stored only in validation truth. Topic consistency missed it and instead scored p06 lowest; lexical consistency flagged p09, with p06 and p05 next lowest. Both judgments are reported rather than selecting whichever looks better.

At a much larger scale, sparse matrices, approximate betweenness, Leiden communities, and approximate nearest-neighbour search would replace dense pairwise operations. The shipped code is a small-study implementation, so the scale change is a design next step, not a tested performance claim.

## E. Next steps outside this brief

A dashboard/API, a real queue and databases, and richer job-title NER could be added after permissions, data rights, and evaluation on an authorised dataset are established. They are not implemented here.
