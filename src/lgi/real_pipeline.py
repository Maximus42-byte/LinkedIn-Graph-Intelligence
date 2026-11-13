"""Isolated real-data pipeline; synthetic committed artifacts are never its targets."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lgi.analysis.career import normalise_seniority
from lgi.analysis.centrality import analyze_centrality
from lgi.analysis.clustering import cluster_profiles
from lgi.analysis.graphs import _add_knn, _combine, build_graphs, write_graph_json
from lgi.analysis.validation import validate_analysis
from lgi.real_tables import write_real_tables
from lgi.secrets import anonymise_profile, load_secrets, pseudonym
from lgi.sources.export import ExportAdapter
from lgi.sources.openalex import OpenAlexAdapter
from lgi.store import write_profiles


def _title_graph(profiles: list[dict], config: dict) -> nx.Graph:
    """Export stubs support title TF-IDF; employer names are separate org evidence."""
    profiles = sorted(profiles, key=lambda row: row["id"])
    ids = [row["id"] for row in profiles]
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    texts = [" ".join(job["title"] for job in row["experience"]).strip() for row in profiles]
    if not any(texts):
        return graph
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([text or "empty" for text in texts])
    except ValueError:
        return graph
    values = cosine_similarity(matrix)
    candidates = {}
    for index, left in enumerate(ids):
        for offset in range(index + 1, len(ids)):
            right = ids[offset]
            weight = float(values[index, offset]) if texts[index] and texts[offset] else 0.0
            candidates[(left, right)] = {"weight": weight, "alternatives": {"tfidf_cosine": weight}}
    _add_knn(graph, ids, candidates, int(config["similarity_knn_k"]), float(config["thresholds"]["semantic"]))
    return graph


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _report(
    path: Path,
    source: str,
    profiles: list[dict],
    bundle: dict,
    centrality: dict,
    clustering: dict,
    validation: dict,
    *,
    anonymised: bool,
    institution: str = "",
    seniority: Counter[str] | None = None,
    connection_ids: set[str] | None = None,
    rendered_tables: dict[str, str],
) -> None:
    graph = bundle["graphs"]["combined_manual"]
    companies = Counter(
        job["company"] for p in profiles if connection_ids is None or p["id"] in connection_ids
        for job in p["experience"] if job["company"]
    )
    community = clustering["labels"].get("louvain", {})
    community_rows = [[pid, label] for pid, label in sorted(community.items())]
    warnings = Counter(warning for profile in profiles for warning in profile["warnings"])
    _csv(path.parent / "tables/source_warnings.csv", ["warning", "profiles"], [
        {"warning": warning, "profiles": count} for warning, count in sorted(warnings.items())
    ])
    _csv(path.parent / "tables/communities.csv", ["id", "louvain_community"], [
        {"id": pid, "louvain_community": label} for pid, label in sorted(community.items())
    ])
    _csv(path.parent / "tables/top_companies.csv", ["company", "profile_records"], [
        {"company": name, "profile_records": count} for name, count in companies.most_common(10)
    ])
    publication_dates = {
        item["post_id"]: item["date"]
        for profile in profiles for item in profile["activity"]
        if item["type"] == "publication"
    }
    stability = validation["perturbation"]
    nulls = validation["null_models"]
    abr = validation["ablation"]
    sensitivity = validation["weight_sensitivity"]
    lines = [
        "# Real-data relationship report",
        "",
        f"Source: {'owner-downloaded LinkedIn export' if source == 'export' else 'official OpenAlex API'}."
        + (f" Institution: {institution}." if institution else ""),
        f"Participants: {len(profiles)}. Combined graph: {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.",
        f"Anonymisation: {'on' if anonymised else 'off'}; {'keyed hashes replace direct identifiers and prose is removed' if anonymised else 'identifiers remain in this local, gitignored output'}.",
        "",
        "## Coverage",
        "",
        rendered_tables["coverage"],
        "",
        "Coverage means at least one identifying record was supplied for a section; optional text is not required. False coverage does not mean the person lacks the attribute.",
        "The text-removal column reports when supplied prose was removed by anonymisation, separately from coverage. Hashed names, employers, institutions and publication IDs still count as identifying fields.",
        "",
        _table(["Source field warning", "Profiles"], [[warning, count] for warning, count in sorted(warnings.items())]),
        "",
        "## Graph and people",
        "",
        rendered_tables["graph_stats"],
        "",
        "Combined graph weights use the configured layer weights and per-layer maximum normalisation.",
        "",
        rendered_tables["key_people"],
        "",
        "## Centrality method agreement",
        "",
        rendered_tables["method_agreement"],
        "",
        _table(["ID", "Louvain community"], community_rows),
        "",
        "For the export, undated shared employers contribute weight 1; no job overlap dates were inferred. Connections are linked to one another when their fields support an edge."
        if source == "export" else
        "Coauthor edges count shared works among the selected authors; the work sample is limited by the configured year and request budget.",
        "",
        "## Role and organisation summaries",
        "",
        _table(["Seniority", "People"], [[name, count] for name, count in sorted((seniority or Counter()).items())]),
        "",
        _table(["Employer or institution", "Profile records"], [[name, count] for name, count in companies.most_common(10)]),
        "",
        "## Clustering and validation",
        "",
        rendered_tables["k_selection"],
        "",
        rendered_tables["clustering_quality"],
        "",
        "All pairwise clustering results are in tables/clustering_pairwise.csv.",
        "These ARI/NMI values compare computed partitions with each other. No real-data ground truth was supplied, so ground-truth ARI/NMI was skipped.",
        "",
        rendered_tables["stability_summary"],
        "",
        rendered_tables["rank_intervals"],
        "",
        rendered_tables["null_models"],
        "",
        rendered_tables["ablation"],
        "",
        f"Edge perturbation: {stability['iterations']} draws; exact top-three retention {stability['top3_set_retention']}; mean overlap {stability['top3_mean_overlap']}; top-one retention {stability['top1_set_retention']}.",
        f"Weight sensitivity: {sensitivity['iterations']} draws; exact top-three retention {sensitivity['top3_set_retention']}; median cluster ARI {sensitivity['cluster_ari_interval'][1] if sensitivity['cluster_ari_interval'] else 'undefined'}.",
        f"Null models: {nulls['degree_preserving']['samples']} degree-preserving samples and {nulls['gnm']['samples']} G(n,m) samples. {nulls['caveat']}",
        f"Ablation covered {len(abr)} layers; values are in validation.json.",
        "",
        "## Limits and skipped analyses",
        "",
    ]
    if source == "export":
        connection_graph = graph.subgraph(connection_ids or set())
        lines.insert(5, f"Connection-only subgraph: {connection_graph.number_of_nodes()} nodes and {connection_graph.number_of_edges()} edges.")
        lines.extend([
            "The owner profile is retained in coverage, while graph analysis uses the connection rows when present.",
            "Connection rows are stubs. Their missing sections cannot support education, skill, activity, career-tenure or post analyses.",
            "Role-title similarity and role-text clustering use supplied titles only when anonymisation is off.",
            "Connection data describes other people and must stay local.",
        ])
    else:
        lines.extend([
            "OpenAlex data is publication metadata, not LinkedIn profile data. Topics are derived, not declared skills; affiliation years are not employment dates.",
            "Education and interests are unavailable. Publications are not reactions and citation counts are not reactions.",
            "The selected author and work sample can omit collaborations outside the request budget and year window.",
            f"Publication dates are copied unchanged from OpenAlex publication_date. Among {len(publication_dates)} distinct selected works, {sum(day.endswith('-01') for day in publication_dates.values())} have day 01. This pipeline cannot determine whether those first-of-month dates are exact or reflect upstream date precision.",
        ])
    lines.append("Skipped analyses and evidence:")
    if anonymised:
        lines.append("- Role and work-title semantic similarity: anonymisation removed titles and other prose. Hashed categories still support exact-overlap comparisons.")
    if not bundle["graphs"]["edu"].number_of_edges():
        lines.append("- Education overlap graph: no pair had usable shared education evidence.")
    if not bundle["graphs"]["activity"].number_of_edges():
        lines.append("- Engagement graph: no participant-to-participant engager IDs were supplied.")
    if source == "export":
        lines.extend([
            "- Career tenure, transitions and talent-source table: connection stubs lack employment dates.",
            "- Skill co-occurrence, skill-gap and bus-factor tables: connection stubs lack declared skills.",
            "- Activity topics, engagement-person table and post/skill consistency: connection stubs lack post activity and declared skills.",
        ])
    else:
        lines.extend([
            "- Job-title seniority, career transitions and talent-source table: OpenAlex supplies affiliation observations, not job titles or employment dates.",
            "- Skill co-occurrence, skill-gap and bus-factor tables: OpenAlex topics are derived labels, not declared skills.",
            "- Activity topics, engagement-person table and post/skill consistency: publications are not social posts, and no participant engagement IDs are supplied.",
        ])
    lines.extend([
        "- Nearest-colleague table and similarity heatmaps: the real pipeline builds graph similarities but does not run the separate talent-matching analysis.",
        "- Synthetic ground-truth table: no real reference labels were supplied.",
    ])
    lines.extend(bundle["warnings"])
    lines.extend(clustering["warnings"])
    lines.extend(["", "This report describes only fields observed in this real-data run. It is separate from the committed synthetic report."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_real_pipeline(args, config: dict) -> None:
    """Process only opt-in local files or the official API into ignored paths."""
    root = Path(config["root"])
    output = root / "output_real"
    profiles_path = root / "data/real_profiles.json"
    if output.is_symlink() or profiles_path.is_symlink():
        raise ValueError("Real output paths must not be symlinks")
    anon = not args.no_anonymise
    secrets_path = Path(args.secrets)
    if not secrets_path.is_absolute():
        secrets_path = root / secrets_path
    key, salt = load_secrets(secrets_path, require_key=args.source == "openalex")
    raw_profiles: list[dict]
    coauthor_counts: dict[tuple[str, str], int] = {}
    institution = ""
    if args.source == "export":
        print("Reminder: connection data describes other people and must stay local.")
        adapter = ExportAdapter(root / "data/real/export")
        raw_profiles = list(adapter.iter_profiles())
        real_config = {**config, "org_ignore_employer_share": float(config["real"]["export_ignore_employer_share"])}
    else:
        adapter = OpenAlexAdapter(
            args.institution, key, root / "data/real/openalex_cache",
            year_from=args.year_from, max_authors=args.max_authors,
            max_requests=args.max_requests, refresh=args.refresh,
        )
        raw_profiles = list(adapter.iter_profiles())
        coauthor_counts = adapter.coauthor_counts
        institution = adapter.institution_name
        real_config = dict(config)
    if not raw_profiles:
        raise ValueError("Real source supplied no profiles")
    graph_raw = raw_profiles[1:] if args.source == "export" and len(raw_profiles) > 1 else raw_profiles
    seniority = Counter(
        normalise_seniority(next(job["title"] for job in profile["experience"] if job["title"]))
        for profile in graph_raw if any(job["title"] for job in profile["experience"])
    )
    profiles = [anonymise_profile(profile, salt) for profile in raw_profiles] if anon else raw_profiles
    if anon and institution:
        institution = pseudonym(salt, "institution", institution)
    id_map = {raw["id"]: safe["id"] for raw, safe in zip(raw_profiles, profiles)}
    connection_ids = {id_map[p["id"]] for p in raw_profiles[1:]} if args.source == "export" else None
    output.mkdir(parents=True, exist_ok=True)
    write_profiles(profiles, profiles_path, output / "profiles.csv")
    graph_profiles = profiles[1:] if args.source == "export" and len(profiles) > 1 else profiles
    bundle = build_graphs(graph_profiles, real_config)
    if args.source == "export":
        bundle["graphs"]["semantic"] = _title_graph(graph_profiles, real_config)
        bundle["embedding_method"] = "TF-IDF on supplied titles" if not anon else "none (titles redacted)"
        for pid in bundle["multilayer"]:
            bundle["multilayer"][pid]["semantic"] = float(bundle["graphs"]["semantic"].degree(pid, weight="weight"))
    weights = {layer: float(config["layer_weights"][layer]) for layer in ("org", "edu", "skill", "semantic", "activity")}
    if args.source == "openalex":
        coauthor = nx.Graph()
        coauthor.add_nodes_from(sorted(p["id"] for p in profiles))
        for (left, right), count in sorted(coauthor_counts.items()):
            coauthor.add_edge(id_map[left], id_map[right], weight=float(count))
        bundle["graphs"]["coauthor"] = coauthor
        bundle["multilayer"] = {
            pid: {**values, "coauthor": float(coauthor.degree(pid, weight="weight"))}
            for pid, values in bundle["multilayer"].items()
        }
        weights["coauthor"] = float(config["real"]["coauthor_weight"])
    bundle["graphs"]["combined_manual"] = _combine(bundle["graphs"], weights, "manual")
    bundle["graphs"]["combined_equal"] = _combine(bundle["graphs"], {layer: 1 / len(weights) for layer in weights}, "equal")
    real_config = {**real_config, "layer_weights": weights, "validation_layers": tuple(weights)}
    centrality = analyze_centrality(bundle["graphs"]["combined_manual"], real_config)
    clustering = cluster_profiles(graph_profiles, bundle, real_config)
    validation = validate_analysis(bundle, centrality, clustering, real_config, None)
    _write_json(output / "validation.json", validation)
    _write_json(output / "clustering.json", clustering)
    write_graph_json(bundle, output / "graph.json", metrics=centrality["metrics"], roles=centrality["roles"], clusters=clustering["labels"])
    rendered_tables = write_real_tables(
        profiles, bundle, centrality, clustering, validation, output,
        anonymised=anon, raw_profiles=raw_profiles,
    )
    _csv(output / "tables/seniority.csv", ["seniority", "people"], [
        {"seniority": label, "people": count} for label, count in sorted(seniority.items())
    ])
    if args.source == "export":
        _csv(output / "tables/connections.csv", ["id", "connected_date"], [
            {"id": p["id"], "connected_date": p.get("connected_date", "")} for p in profiles[1:]
        ])
    else:
        (output / "tables/connections.csv").unlink(missing_ok=True)
    _report(output / "report.md", args.source, profiles, bundle, centrality, clustering, validation,
            anonymised=anon, institution=institution, seniority=seniority,
            connection_ids=connection_ids, rendered_tables=rendered_tables)
    print(f"Processed {len(profiles)} real-source profiles; outputs are in {output} and {profiles_path}")
