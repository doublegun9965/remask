from common.parsing.parse_and_get_acc import extract_gsm_answer


def _as_float(value):
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").replace("$", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def score_gsm8k_generations(generations: list[dict], annotate: bool = False) -> dict:
    """Compute numeric exact-match accuracy for saved GSM8K generations."""
    correct = 0
    parsed = 0
    nfes = []
    remask_counts = []

    for item in generations:
        prediction = extract_gsm_answer(item.get("generations", ""))
        ground_truth = _as_float(item.get("ground_truth"))
        if prediction is not None:
            parsed += 1
        is_correct = (
            prediction is not None
            and ground_truth is not None
            and abs(prediction - ground_truth) < 1e-6
        )
        correct += int(is_correct)
        if item.get("steps") is not None:
            nfes.append(float(item["steps"]))
        if item.get("remasks") is not None:
            remask_counts.append(float(item["remasks"]))

        if annotate:
            item["extracted_answer"] = prediction
            item["is_correct"] = is_correct

    total = len(generations)
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "parsed_answers": parsed,
        "evaluated": total,
        "average_nfe": sum(nfes) / len(nfes) if nfes else 0.0,
        "total_nfe": sum(nfes),
        "min_nfe": min(nfes) if nfes else 0.0,
        "max_nfe": max(nfes) if nfes else 0.0,
        "average_remasks": (
            sum(remask_counts) / len(remask_counts) if remask_counts else 0.0
        ),
        "total_remasks": sum(remask_counts),
    }
    return metrics
