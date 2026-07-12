"""
RAG Output Quality Benchmarking Suite
Evaluates retrieval-augmented generation pipelines across
five quality dimensions with automated scoring and reporting.
"""

import json
import os
import re
import math
from datetime import datetime
from collections import defaultdict


def load_test_cases(filepath="test_cases.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def compute_rouge_l(hypothesis, reference):
    """
    Computes ROUGE-L score using longest common subsequence.
    Measures sequence-level similarity between generated
    response and ground truth.
    """
    h_tokens = tokenize(hypothesis)
    r_tokens = tokenize(reference)

    if not h_tokens or not r_tokens:
        return 0.0

    m, n = len(h_tokens), len(r_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if h_tokens[i-1] == r_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs = dp[m][n]
    precision = lcs / m if m > 0 else 0
    recall = lcs / n if n > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    return round(f1, 4)


def compute_token_overlap(text_a, text_b):
    """
    Computes token-level overlap between two texts.
    Used for context grounding and faithfulness scoring.
    """
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return round(len(intersection) / len(union), 4)


def compute_cosine_similarity_simple(text_a, text_b):
    """
    Lightweight TF-based cosine similarity without
    external embedding models — suitable for offline evaluation.
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    vocab = set(tokens_a) | set(tokens_b)
    if not vocab:
        return 0.0

    vec_a = {w: tokens_a.count(w) for w in vocab}
    vec_b = {w: tokens_b.count(w) for w in vocab}

    dot = sum(vec_a[w] * vec_b[w] for w in vocab)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return round(dot / (mag_a * mag_b), 4)


def score_faithfulness(response, context):
    """
    Measures how grounded the response is in the
    retrieved context. High faithfulness means the
    response stays within the boundaries of provided
    context without introducing external claims.
    """
    overlap = compute_token_overlap(response, context)
    cosine = compute_cosine_similarity_simple(response, context)

    score = (overlap * 0.5 + cosine * 0.5) * 100
    return round(score, 2)


def score_answer_relevance(response, query):
    """
    Measures how directly the response addresses
    the original query. Low relevance indicates
    topic drift or incomplete query resolution.
    """
    cosine = compute_cosine_similarity_simple(response, query)
    overlap = compute_token_overlap(response, query)

    score = (cosine * 0.7 + overlap * 0.3) * 100
    return round(min(score * 1.5, 100), 2)


def score_context_precision(context, query):
    """
    Evaluates whether retrieved context is relevant
    to the query. Low precision indicates retrieval
    pipeline is fetching irrelevant chunks.
    """
    cosine = compute_cosine_similarity_simple(context, query)
    score = cosine * 100
    return round(min(score * 1.8, 100), 2)


def score_groundedness(response, context):
    """
    Detects hallucination risk by identifying response
    claims not supported by retrieved context.
    Penalises absolute language and unsourced specifics.
    """
    base_score = compute_token_overlap(response, context) * 100

    hallucination_markers = [
        "always", "never", "everyone", "no one",
        "100%", "guaranteed", "definitely", "certainly",
        "all scientists", "all experts", "proven fact"
    ]

    response_lower = response.lower()
    penalty = sum(
        10 for marker in hallucination_markers
        if marker in response_lower
    )

    score = max(0, base_score - penalty)
    return round(min(score, 100), 2)


def score_rouge(response, ground_truth):
    """
    ROUGE-L based similarity against ground truth.
    Measures sequence-level quality of the response
    against expected output.
    """
    rouge_l = compute_rouge_l(response, ground_truth)
    return round(rouge_l * 100, 2)


def classify_result(score):
    if score >= 75:
        return "PASS"
    elif score >= 55:
        return "REVIEW"
    else:
        return "FAIL"


def evaluate_test_case(tc):
    """
    Runs all five evaluation dimensions against
    a single test case and returns structured result.
    """
    query = tc["query"]
    context = tc["retrieved_context"]
    response = tc["generated_response"]
    ground_truth = tc["ground_truth"]

    scores = {
        "faithfulness": score_faithfulness(response, context),
        "answer_relevance": score_answer_relevance(response, query),
        "context_precision": score_context_precision(context, query),
        "groundedness": score_groundedness(response, context),
        "rouge_l": score_rouge(response, ground_truth)
    }

    weighted_overall = round(
        scores["faithfulness"] * 0.30 +
        scores["answer_relevance"] * 0.25 +
        scores["context_precision"] * 0.15 +
        scores["groundedness"] * 0.20 +
        scores["rouge_l"] * 0.10,
        2
    )

    dimension_results = {
        dim: {
            "score": score,
            "status": classify_result(score)
        }
        for dim, score in scores.items()
    }

    flags = [
        dim for dim, result in dimension_results.items()
        if result["status"] != "PASS"
    ]

    return {
        "test_case_id": tc["id"],
        "category": tc["category"],
        "query_preview": query[:80],
        "overall_score": weighted_overall,
        "overall_status": classify_result(weighted_overall),
        "dimensions": dimension_results,
        "flags": flags,
        "pass_rate": round(
            len([d for d in dimension_results.values()
                 if d["status"] == "PASS"]) / len(dimension_results) * 100,
            1
        )
    }


def generate_benchmark_report(suite_data, results):
    """
    Aggregates all test case results into a
    pipeline-level benchmark report with summary
    statistics and dimension-level analysis.
    """
    overall_scores = [r["overall_score"] for r in results]
    avg_score = round(sum(overall_scores) / len(overall_scores), 2)

    dimension_averages = {}
    for dim in ["faithfulness", "answer_relevance",
                "context_precision", "groundedness", "rouge_l"]:
        scores = [r["dimensions"][dim]["score"] for r in results]
        dimension_averages[dim] = round(
            sum(scores) / len(scores), 2
        )

    category_breakdown = defaultdict(list)
    for r in results:
        category_breakdown[r["category"]].append(r["overall_score"])

    category_averages = {
        cat: round(sum(scores) / len(scores), 2)
        for cat, scores in category_breakdown.items()
    }

    status_counts = defaultdict(int)
    for r in results:
        status_counts[r["overall_status"]] += 1

    weakest_dimension = min(
        dimension_averages, key=dimension_averages.get
    )
    strongest_dimension = max(
        dimension_averages, key=dimension_averages.get
    )

    return {
        "report_metadata": {
            "suite": suite_data["benchmark_suite"],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_cases": len(results),
            "evaluation_dimensions": 5
        },
        "pipeline_summary": {
            "average_overall_score": avg_score,
            "pipeline_status": classify_result(avg_score),
            "pass_count": status_counts.get("PASS", 0),
            "review_count": status_counts.get("REVIEW", 0),
            "fail_count": status_counts.get("FAIL", 0),
            "pass_rate_percent": round(
                status_counts.get("PASS", 0) / len(results) * 100, 1
            )
        },
        "dimension_averages": dimension_averages,
        "weakest_dimension": weakest_dimension,
        "strongest_dimension": strongest_dimension,
        "category_performance": category_averages,
        "recommendations": generate_recommendations(
            dimension_averages, weakest_dimension
        ),
        "test_case_results": results
    }


def generate_recommendations(dimension_averages, weakest):
    """
    Produces actionable pipeline improvement recommendations
    based on dimension-level performance patterns.
    """
    recommendations = []

    thresholds = {
        "faithfulness": (
            "Retrieval pipeline returning insufficient context. "
            "Review chunk size and overlap parameters."
        ),
        "answer_relevance": (
            "Response generation drifting from query intent. "
            "Tighten system prompt to enforce query-focused responses."
        ),
        "context_precision": (
            "Retriever fetching low-relevance chunks. "
            "Review embedding model and similarity threshold."
        ),
        "groundedness": (
            "Hallucination risk elevated. Add factual grounding "
            "instructions to prompt. Consider adding a verification step."
        ),
        "rouge_l": (
            "Response surface form deviates from expected output. "
            "Review output formatting instructions in prompt."
        )
    }

    for dim, score in dimension_averages.items():
        if score < 75:
            recommendations.append({
                "dimension": dim,
                "score": score,
                "priority": "HIGH" if score < 55 else "MEDIUM",
                "action": thresholds.get(dim, "Manual review required.")
            })

    return sorted(
        recommendations,
        key=lambda x: x["score"]
    )


def print_report(report):
    print("\n" + "=" * 65)
    print("RAG PIPELINE QUALITY BENCHMARK REPORT")
    print("=" * 65)
    print(f"Suite:      {report['report_metadata']['suite']}")
    print(f"Generated:  {report['report_metadata']['generated_at']}")
    print(f"Test Cases: {report['report_metadata']['total_test_cases']}")
    print("-" * 65)
    print("PIPELINE SUMMARY")
    print("-" * 65)
    ps = report["pipeline_summary"]
    print(f"Average Score:    {ps['average_overall_score']}%")
    print(f"Pipeline Status:  {ps['pipeline_status']}")
    print(f"Pass:  {ps['pass_count']}  |  "
          f"Review: {ps['review_count']}  |  "
          f"Fail: {ps['fail_count']}")
    print(f"Overall Pass Rate: {ps['pass_rate_percent']}%")
    print("-" * 65)
    print("DIMENSION AVERAGES")
    print("-" * 65)
    for dim, score in report["dimension_averages"].items():
        bar = "█" * int(score / 5)
        print(f"{dim.replace('_', ' '):<22} {score:>6}%  {bar}")
    print(f"\nStrongest: {report['strongest_dimension'].replace('_', ' ')}")
    print(f"Weakest:   {report['weakest_dimension'].replace('_', ' ')}")
    print("-" * 65)
    print("CATEGORY PERFORMANCE")
    print("-" * 65)
    for cat, score in report["category_performance"].items():
        status = classify_result(score)
        print(f"{cat.replace('_', ' '):<25} {score}%  [{status}]")
    print("-" * 65)
    print("TEST CASE RESULTS")
    print("-" * 65)
    for r in report["test_case_results"]:
        print(f"\n{r['test_case_id']} [{r['category']}]")
        print(f"  Query:   {r['query_preview']}...")
        print(f"  Score:   {r['overall_score']}%  "
              f"Status: {r['overall_status']}  "
              f"Pass rate: {r['pass_rate']}%")
        if r["flags"]:
            print(f"  Flags:   {', '.join(r['flags'])}")
    print("-" * 65)
    if report["recommendations"]:
        print("RECOMMENDATIONS")
        print("-" * 65)
        for rec in report["recommendations"]:
            print(f"\n[{rec['priority']}] "
                  f"{rec['dimension'].replace('_', ' ').upper()} "
                  f"— {rec['score']}%")
            print(f"  {rec['action']}")
    print("=" * 65)


def save_report(report, filepath="benchmark_report.json"):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {filepath}")


if __name__ == "__main__":
    suite_data = load_test_cases()
    print(f"Loaded {len(suite_data['test_cases'])} test cases")
    print(f"Suite: {suite_data['benchmark_suite']}\n")

    results = []
    for tc in suite_data["test_cases"]:
        result = evaluate_test_case(tc)
        results.append(result)
        print(f"{result['test_case_id']} — "
              f"{result['overall_score']}% — "
              f"{result['overall_status']}")

    report = generate_benchmark_report(suite_data, results)
    print_report(report)
    save_report(report)
