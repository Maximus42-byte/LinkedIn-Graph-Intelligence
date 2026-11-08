"""Small offline checks for the phase 7 draft modules."""

import json

import networkx as nx

from lgi.analysis.activity import analyze_activity
from lgi.analysis.career import analyze_career, normalise_seniority
from lgi.analysis.similarity import analyze_similarity


def profiles():
    return [
        {
            "id": "p01", "basic": {"headline": "ML Engineer", "summary": "machine learning models"},
            "experience": [
                {"company": "Fictional A", "title": "Junior Engineer", "start": "2020-01", "end": "2021-01"},
                {"company": "Fictional B", "title": "Senior ML Engineer", "start": "2021-02", "end": None},
            ],
            "education": [{"school": "Fictional U", "field": "AI"}],
            "skills": ["Python", "Machine Learning"],
            "activity": [{"post_id": "a", "date": "2024-03-02", "text": "Machine learning models and Python", "reaction_count": 4, "engaged_ids": ["p02"]}],
        },
        {
            "id": "p02", "basic": {"headline": "Data Scientist", "summary": "data models"},
            "experience": [
                {"company": "Fictional A", "title": "Data Scientist", "start": "2020-04", "end": "2022-04"},
                {"company": "Fictional B", "title": "Lead Scientist", "start": "2022-05", "end": None},
            ],
            "education": [{"school": "Fictional U", "field": "AI"}],
            "skills": ["Python", "Statistics"],
            "activity": [{"post_id": "b", "date": "2024-02-02", "text": "Python statistics models", "reaction_count": 3, "engaged_ids": ["p01"]}],
        },
        {
            "id": "p03", "basic": {"headline": "Security Analyst", "summary": "security systems"},
            "experience": [{"company": "Fictional C", "title": "Security Analyst", "start": "2019-01", "end": None}],
            "education": [], "skills": ["Security", "Threat Modeling"],
            "activity": [{"post_id": "c", "date": "2024-01-02", "text": "Threat security systems", "reaction_count": 1, "engaged_ids": []}],
        },
    ]


def test_career_transitions_and_skill_coverage():
    result = analyze_career(profiles(), "2024-03")
    assert normalise_seniority("Senior Engineering Manager") == "manager/director"
    assert result["as_of_month"] == "2024-03"
    later = analyze_career(profiles(), "2026-01")
    assert later["as_of_month"] == "2026-01"
    assert later["roles"][1]["tenure_months"] > result["roles"][1]["tenure_months"]
    assert result["people"][0]["transition_count"] == 1
    assert result["education_schools"] == [{"school": "Fictional U", "profile_count": 2}]
    assert {row["skill"] for row in result["bus_factor_one_skills"]} == {"machine learning", "statistics", "security", "threat modeling"}
    assert result["movement_graph"]["directed"] is True


def test_activity_topics_and_graph_comparison():
    org = nx.Graph()
    org.add_nodes_from(["p01", "p02", "p03"])
    org.add_edge("p01", "p02", weight=5)
    result = analyze_activity(profiles(), {"org": org}, {"seed": 73, "activity_topics": 2})
    assert len(result["topics"]) == 2
    assert result["graph_comparison"]["org_engagement_edge_jaccard"] == 1.0
    assert result["people"][0]["topic_consistency"] is not None
    assert result["people"][0]["engagement_weighted_degree"] == 2.0


def test_similarity_is_symmetric_and_has_no_self_edges():
    result = analyze_similarity(profiles(), {"knn_k": 2})
    assert result["ids"] == ["p01", "p02", "p03"]
    assert result["jaccard_matrix"][0][1] == result["jaccard_matrix"][1][0]
    assert all(edge["source"] != edge["target"] for edge in result["knn_graph"]["edges"])
    json.dumps(result, sort_keys=True, allow_nan=False)


def test_missing_data_stays_serializable():
    result = {
        "career": analyze_career([], "2024-03"),
        "activity": analyze_activity([], None),
        "similarity": analyze_similarity([], {"knn_k": 3}),
    }
    json.dumps(result, sort_keys=True, allow_nan=False)
