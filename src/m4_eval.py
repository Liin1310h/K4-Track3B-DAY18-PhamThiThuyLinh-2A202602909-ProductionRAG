from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}
    try:
        import ragas as _ragas
        from ragas import evaluate
        from datasets import Dataset

        # Ensure OPENAI_API_KEY is in environment (ragas 0.2 uses langchain_openai)
        from config import OPENAI_API_KEY as _key
        if _key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = _key

        _ver = tuple(int(x) for x in _ragas.__version__.split(".")[:2])
        if _ver >= (0, 2):
            from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
            metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
            # ragas 0.2 renames columns: question→user_input, answer→response, ground_truth→reference
            ds_dict = {
                "user_input": questions,
                "response": answers,
                "retrieved_contexts": contexts,
                "reference": ground_truths,
            }
            q_col, a_col, c_col, gt_col = "user_input", "response", "retrieved_contexts", "reference"
            f_col, ar_col, cp_col, cr_col = "faithfulness", "answer_relevancy", "context_precision", "context_recall"
        else:
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            ds_dict = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
            q_col, a_col, c_col, gt_col = "question", "answer", "contexts", "ground_truth"
            f_col, ar_col, cp_col, cr_col = "faithfulness", "answer_relevancy", "context_precision", "context_recall"

        dataset = Dataset.from_dict(ds_dict)
        result = evaluate(dataset, metrics=metrics)
        df = result.to_pandas()

        def _get(row, key):
            # find actual column (ragas may suffix with score)
            for col in [key, key + "_score"]:
                if col in row:
                    return float(row[col] or 0.0)
            return 0.0

        per_question = [
            EvalResult(
                question=row.get(q_col, ""),
                answer=row.get(a_col, ""),
                contexts=row.get(c_col, []),
                ground_truth=row.get(gt_col, ""),
                faithfulness=_get(row, f_col),
                answer_relevancy=_get(row, ar_col),
                context_precision=_get(row, cp_col),
                context_recall=_get(row, cr_col),
            )
            for _, row in df.iterrows()
        ]

        def _mean(col):
            for c in [col, col + "_score"]:
                if c in df.columns:
                    return float(df[c].mean())
            return 0.0

        return {
            "faithfulness": _mean(f_col),
            "answer_relevancy": _mean(ar_col),
            "context_precision": _mean(cp_col),
            "context_recall": _mean(cr_col),
            "per_question": [vars(r) for r in per_question],
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored = []
    for r in eval_results:
        # accept both EvalResult dataclass and plain dict
        if isinstance(r, dict):
            get = lambda k: r.get(k, 0.0) or 0.0  # noqa: E731
        else:
            get = lambda k: getattr(r, k, 0.0) or 0.0  # noqa: E731
        metrics = {
            "faithfulness": get("faithfulness"),
            "context_recall": get("context_recall"),
            "context_precision": get("context_precision"),
            "answer_relevancy": get("answer_relevancy"),
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=lambda k: metrics[k])
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        question = r.get("question", "") if isinstance(r, dict) else getattr(r, "question", "")
        scored.append({
            "question": question,
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "avg_score": round(avg, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    scored.sort(key=lambda x: x["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
