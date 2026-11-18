"""资格证据与不确定性。

资格只由档案中的现行评分推导；历史决定作为既成事实保留，但不会被
当成独立真值，也不覆盖重新推导的结果。每个技能给出 ``met`` /
``not_met`` / ``uncertain`` 三种口径，并保留评分编号，便于追溯作品、
评分者和时间。

除估计值外还给出“比较性解释”：某学员的能力区间与所在评分组的中位
学员相比是分开还是重叠，帮助使用者区分个人能力差异和尺度噪声。
"""

STATUS_LABELS = {
    "met": "达到",
    "not_met": "未达到",
    "uncertain": "不确定",
}

OUTCOME_LABELS = {
    "qualified": "证据达到要求",
    "not_qualified": "证据未达到要求",
    "uncertain": "证据不确定",
}

# 供汇总使用的稳定原因码；人类可读说明另见 reasons。
REASON_LABELS = {
    "no_network": "该技能尚无评分网络",
    "insufficient_judgments": "评分条数或作品件数不足",
    "scale_inseparable": "评分组无法分离学员能力与评分者尺度",
    "not_converged": "能力估计未收敛",
    "interval_straddles": "能力区间跨过岗位要求",
    "below_requirement": "能力区间整体低于岗位要求",
}


def qualification_evidence(dataset, networks, student_id, vacancy_id):
    """汇总学员对某岗位必要技能的证据，并附带可追溯的评分编号。"""
    vacancy = dataset.vacancies[vacancy_id]
    skills = []
    reasons = []
    codes = set()
    for requirement in vacancy["requirements"]:
        skill_id = requirement["skill_id"]
        network = networks.get(skill_id)
        if network is None:
            detail = {
                "skill_id": skill_id,
                "required_level": requirement["level"],
                "status": "uncertain",
                "estimates": None,
                "comparison": None,
                "reason": REASON_LABELS["no_network"],
                "reason_code": "no_network",
            }
        else:
            status, estimate = network.required_level(student_id, requirement["level"])
            code = _reason_code(status, estimate)
            detail = {
                "skill_id": skill_id,
                "required_level": requirement["level"],
                "status": status,
                "estimates": _estimate_summary(estimate),
                "comparison": _comparison(network, student_id, estimate),
                "reason": _skill_reason(requirement, status, estimate),
                "reason_code": code,
            }
        skills.append(detail)
        if detail["status"] != "met":
            reasons.append(f"{skill_id}：{detail['reason']}")
            codes.add(detail["reason_code"])
    if any(skill["status"] == "not_met" for skill in skills):
        outcome = "not_qualified"
    elif any(skill["status"] == "uncertain" for skill in skills):
        outcome = "uncertain"
    else:
        outcome = "qualified"
    rating_ids = sorted({rating_id for skill in skills if skill["estimates"]
                         for rating_id in skill["estimates"]["rating_ids"]})
    return {
        "student_id": student_id,
        "vacancy_id": vacancy_id,
        "outcome": outcome,
        "skills": skills,
        "reasons": reasons,
        "reason_codes": sorted(codes),
        "rating_ids": rating_ids,
        "component_ids": sorted({skill["estimates"]["component_id"]
                                 for skill in skills if skill["estimates"]
                                 and skill["estimates"]["component_id"]}),
        "provenance": [_rating_provenance(dataset, rating_id)
                       for rating_id in rating_ids],
    }


def _reason_code(status, estimate):
    if estimate["confidence"] != "usable":
        return estimate["evidence_code"] or "insufficient_judgments"
    if status == "not_met":
        return "below_requirement"
    return "interval_straddles"


def _skill_reason(requirement, status, estimate):
    level = requirement["level"]
    if estimate["confidence"] != "usable":
        return estimate["evidence"]
    if status == "met":
        return (f"能力区间 {_fmt_interval(estimate['interval'])} 整体高于要求 "
                f"{level:+.2f}")
    if status == "not_met":
        return (f"能力区间 {_fmt_interval(estimate['interval'])} 整体低于要求 "
                f"{level:+.2f}")
    return (f"能力区间 {_fmt_interval(estimate['interval'])} 跨过要求 {level:+.2f}，"
            f"证据不足以判断是否达到")


def _fmt_interval(interval):
    if not interval:
        return "无"
    return f"[{interval[0]:+.2f}, {interval[1]:+.2f}]"



def _comparison(network, student_id, estimate):
    """说明该学员的区间在本评分组中与均值是否分开。"""
    component = network.locate(student_id)
    if component is None or estimate["estimate"] is None:
        return None
    low, high = estimate["interval"]
    others = sorted(component["ability"].values())
    middle = others[len(others) // 2] if others else 0.0
    if low > middle:
        position = "高于组内中位学员"
    elif high < middle:
        position = "低于组内中位学员"
    else:
        position = "与组内中位学员不可区分"
    return {
        "component_id": component["component_id"],
        "component_median": middle,
        "position": position,
        "interval_separated_from_median": low > middle or high < middle,
    }


def _estimate_summary(estimate):
    return {
        "component_id": estimate["component_id"],
        "value": estimate["estimate"],
        "standard_error": estimate["standard_error"],
        "interval": estimate["interval"],
        "evidence": estimate["evidence"],
        "confidence": estimate["confidence"],
        "rating_ids": estimate["rating_ids"],
        "work_ids": estimate["work_ids"],
        "rater_ids": estimate["rater_ids"],
    }


def _rating_provenance(dataset, rating_id):
    rating = dataset.ratings[rating_id]
    return {
        "rating_id": rating_id,
        "work_id": rating["work_id"],
        "skill_id": rating["skill_id"],
        "rater_id": rating["rater_id"],
        "score": rating["score"],
        "rated_at": rating["rated_at"],
        "supersedes": rating.get("replaces_rating_id") or None,
        "superseded_by": dataset.superseded_by.get(rating_id),
        "current": rating_id in dataset.current_rating_ids,
    }


def historic_decisions(dataset, student_id=None, vacancy_id=None):
    """列出决定及其依据，说明哪些依据已被重评替换。"""
    result = []
    for decision in dataset.decisions.values():
        if student_id and decision["student_id"] != student_id:
            continue
        if vacancy_id and decision["vacancy_id"] != vacancy_id:
            continue
        chains = []
        for rating_id in decision["evidence_rating_ids"]:
            chain = dataset.rating_chain(rating_id)
            chains.append({
                "rating_id": rating_id,
                "still_current": chain == [rating_id],
                "chain": chain,
            })
        result.append({
            "decision_id": decision["decision_id"],
            "student_id": decision["student_id"],
            "vacancy_id": decision["vacancy_id"],
            "decided_at": decision["decided_at"],
            "outcome": decision["outcome"],
            "reason": decision["reason"],
            "evidence": chains,
            "estimates": decision["estimates"],
        })
    return sorted(result, key=lambda row: (row["decided_at"], row["decision_id"]))


def confirmed_standing(dataset, networks, registration):
    """对照已确定报名的原依据与重新推导的资格，不修改任何既成事实。"""
    decision = dataset.decisions[registration["decision_id"]]
    superseded = []
    for rating_id in decision["evidence_rating_ids"]:
        chain = dataset.rating_chain(rating_id)
        if len(chain) > 1:
            superseded.append({"rating_id": rating_id, "chain": chain})
    current = qualification_evidence(dataset, networks, registration["student_id"],
                                     registration["vacancy_id"])
    return {
        "registration_id": registration["registration_id"],
        "decision_id": decision["decision_id"],
        "outcome": decision["outcome"],
        "reason": decision["reason"],
        "kept": True,
        "note": "已确定资格保持不变，不因后续评分被自动撤销。",
        "revised_evidence": superseded,
        "current_recommendation": current["outcome"],
        "current_rating_ids": current["rating_ids"],
        "current_provenance": current["provenance"],
        "current_reasons": current["reasons"],
    }


def trace_decision(dataset, decision_id):
    """逐条列出决定依据评分是否仍为现行证据，以及被哪条重评接替。"""
    decision = dataset.decisions[decision_id]
    rows = []
    for rating_id in decision["evidence_rating_ids"]:
        chain = dataset.rating_chain(rating_id)
        rating = dataset.ratings[rating_id]
        rows.append({
            "rating_id": rating_id,
            "work_id": rating["work_id"],
            "skill_id": rating["skill_id"],
            "rater_id": rating["rater_id"],
            "score": rating["score"],
            "rated_at": rating["rated_at"],
            "still_current": chain == [rating_id],
            "replaces": rating.get("replaces_rating_id") or None,
            "superseded_by": dataset.superseded_by.get(rating_id),
            "chain": chain,
        })
    return {
        "decision_id": decision_id,
        "student_id": decision["student_id"],
        "vacancy_id": decision["vacancy_id"],
        "decided_at": decision["decided_at"],
        "outcome": decision["outcome"],
        "reason": decision["reason"],
        "evidence": rows,
        "estimates": decision["estimates"],
        "contains_superseded_evidence": any(not row["still_current"] for row in rows),
    }


def describe(evidence):
    return OUTCOME_LABELS.get(evidence.get("outcome"), str(evidence.get("outcome")))


__all__ = ["STATUS_LABELS", "OUTCOME_LABELS", "qualification_evidence",
           "historic_decisions", "confirmed_standing", "trace_decision", "describe"]
