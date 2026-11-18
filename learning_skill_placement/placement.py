"""岗位推荐与带教负荷。

推荐把资格证据和时段容量放在一起核算：

* ``independent``：证据达到要求，且该时段的独立席位仍有空余。
* ``supervised``：占用某带教者在同一时段簇中的带教名额。
* ``pending``：证据不确定、缺评或没有可用容量，逐条写明暂缓原因。

带教容量同时受两个上限约束：时段自身的带教席位，以及带教者在所有
时间重叠服务中的同时带教人数（用扫描线求峰值，而非简单两两比较）。
两者都按“同时发生”核算，不按时段逐条孤立判断，也不允许超卖。

已确定报名先占容量、原样保留，不因新证据被撤销，也不被重新分配。
"""

from collections import defaultdict

from .qualification import confirmed_standing, qualification_evidence


def _overlaps(left, right):
    if left["date"] != right["date"]:
        return False
    return left["start"] < right["end"] and right["start"] < left["end"]


def sessions_overlap(dataset, left_id, right_id):
    return _overlaps(dataset.sessions[left_id], dataset.sessions[right_id])


def mentor_of(dataset, registration):
    return (registration["mentor_id"]
            or dataset.sessions[registration["session_id"]]["mentor_id"])


def max_concurrent(sessions):
    """按扫描线求一组服务时段在同一时刻的最大并发数。"""
    events = defaultdict(int)
    for session in sessions:
        events[(session["date"], session["start"])] += 1
        events[(session["date"], session["end"])] -= 1
    running = peak = 0
    for key in sorted(events):
        running += events[key]
        peak = max(peak, running)
    return peak


def _supervised_used(supervised_sessions, session_id):
    return sum(1 for assigned in supervised_sessions if assigned["session_id"] == session_id)


REASON_LABELS = {
    "evidence_uncertain": "资格证据不确定",
    "evidence_below_requirement": "必要技能证据未达到要求",
    "overlapping_commitment": "学员在重叠时段已有已确定服务",
    "independent_full": "独立席位已满",
    "no_mentor": "该时段没有可用的带教者",
    "supervised_slots_full": "时段带教席位已满",
    "mentor_capacity_full": "带教者在重叠时段名额已满",
}


def build_plan(dataset, networks):
    """审核待定报名并生成逐条推荐；已确定报名只计入容量，不被改写。"""
    confirmed = sorted((row for row in dataset.registrations.values()
                        if row["status"] == "confirmed"),
                       key=lambda row: row["registration_id"])
    requested = sorted((row for row in dataset.registrations.values()
                        if row["status"] == "requested"),
                       key=lambda row: row["registration_id"])

    independent_used = defaultdict(int)
    mentor_assignments = defaultdict(list)
    supervised_sessions = []
    student_sessions = defaultdict(list)
    for registration in confirmed:
        session = dataset.sessions[registration["session_id"]]
        student_sessions[registration["student_id"]].append(session)
        if registration["mode"] == "independent":
            independent_used[registration["session_id"]] += 1
        else:
            mentor_id = mentor_of(dataset, registration)
            mentor_assignments[mentor_id].append(session)
            supervised_sessions.append({"session_id": registration["session_id"],
                                        "mentor_id": mentor_id})
    retained = [confirmed_standing(dataset, networks, row) for row in confirmed]

    plans = []
    for registration in requested:
        session = dataset.sessions[registration["session_id"]]
        qualification = qualification_evidence(
            dataset, networks, registration["student_id"], registration["vacancy_id"])
        entry = {
            "registration_id": registration["registration_id"],
            "student_id": registration["student_id"],
            "vacancy_id": registration["vacancy_id"],
            "session_id": registration["session_id"],
            "requested_mode": registration["mode"],
            "qualification": qualification["outcome"],
            "recommendation": "pending",
            "mentor_id": None,
            "reasons": list(qualification["reasons"]),
            "reason_codes": list(qualification["reason_codes"]),
            "rating_ids": qualification["rating_ids"],
        }
        if any(_overlaps(session, other)
               for other in student_sessions[registration["student_id"]]):
            entry["reasons"].append(REASON_LABELS["overlapping_commitment"])
            entry["reason_codes"].append("overlapping_commitment")
            plans.append(entry)
            continue
        if qualification["outcome"] != "qualified":
            if qualification["outcome"] == "uncertain":
                entry["reasons"].append(
                    "证据不确定，暂缓判断，不能承诺独立上岗")
                entry["reason_codes"].append("evidence_uncertain")
            else:
                entry["reasons"].append(
                    REASON_LABELS["evidence_below_requirement"])
                entry["reason_codes"].append("evidence_below_requirement")
            plans.append(entry)
            continue

        if (registration["mode"] != "supervised"
                and independent_used[registration["session_id"]]
                < session["independent_slots"]):
            independent_used[registration["session_id"]] += 1
            entry["recommendation"] = "independent"
            student_sessions[registration["student_id"]].append(session)
            plans.append(entry)
            continue

        if registration["mode"] != "supervised":
            entry["reasons"].append(REASON_LABELS["independent_full"])
            entry["reason_codes"].append("independent_full")

        mentor_id = mentor_of(dataset, registration)
        if mentor_id is None:
            entry["reasons"].append(REASON_LABELS["no_mentor"])
            entry["reason_codes"].append("no_mentor")
            plans.append(entry)
            continue
        capacity = dataset.mentors[mentor_id]["capacity"]
        peak = max_concurrent(list(mentor_assignments[mentor_id]) + [session])
        session_used = _supervised_used(supervised_sessions, session["session_id"])
        if session_used >= session["supervised_slots"]:
            entry["reasons"].append(
                f"时段 {session['session_id']} 的带教席位已满"
                f"（{session_used}/{session['supervised_slots']}）")
            entry["reason_codes"].append("supervised_slots_full")
            plans.append(entry)
            continue
        if peak > capacity:
            entry["reasons"].append(
                f"带教者 {mentor_id} 在重叠时段的同时带教人数会达到 "
                f"{peak}，超出容量 {capacity}")
            entry["reason_codes"].append("mentor_capacity_full")
            plans.append(entry)
            continue
        entry["recommendation"] = "supervised"
        entry["mentor_id"] = mentor_id
        mentor_assignments[mentor_id].append(session)
        supervised_sessions.append({"session_id": session["session_id"],
                                    "mentor_id": mentor_id})
        student_sessions[registration["student_id"]].append(session)
        plans.append(entry)

    return {
        "requested": plans,
        "retained": retained,
        "mentor_assignments": {key: list(value)
                               for key, value in mentor_assignments.items()},
        "independent_used": dict(independent_used),
    }


def summarize_plan(dataset, networks):
    """按岗位、带教者和暂缓原因汇总，方便横向比较覆盖与负担。"""
    plan = build_plan(dataset, networks)
    per_vacancy = {}
    for session in dataset.sessions.values():
        row = per_vacancy.setdefault(session["vacancy_id"], {
            "vacancy_id": session["vacancy_id"],
            "sessions": 0, "capacity": 0, "confirmed": 0,
            "independent": 0, "supervised": 0, "pending": 0,
            "unfilled_slots": 0,
        })
        row["sessions"] += 1
        row["capacity"] += session["independent_slots"] + session["supervised_slots"]
    for registration in dataset.registrations.values():
        if registration["status"] == "confirmed":
            per_vacancy[registration["vacancy_id"]]["confirmed"] += 1
    for entry in plan["requested"]:
        row = per_vacancy.setdefault(entry["vacancy_id"], {
            "vacancy_id": entry["vacancy_id"], "sessions": 0, "capacity": 0,
            "confirmed": 0, "independent": 0, "supervised": 0, "pending": 0,
            "unfilled_slots": 0})
        row[entry["recommendation"]] += 1
    for row in per_vacancy.values():
        row["fillable_slots"] = row["capacity"] - row["confirmed"]
        row["unfilled_slots"] = max(row["fillable_slots"]
                                    - row["independent"] - row["supervised"], 0)
        row["over_capacity"] = (row["confirmed"] + row["independent"]
                                + row["supervised"]) > row["capacity"]

    mentor_rows = []
    for mentor_id, mentor in sorted(dataset.mentors.items()):
        assignments = plan["mentor_assignments"].get(mentor_id, [])
        peak = max_concurrent(assignments)
        mentor_rows.append({
            "mentor_id": mentor_id,
            "capacity": mentor["capacity"],
            "assigned": len(assignments),
            "peak_concurrent": peak,
            "spare_concurrent": max(mentor["capacity"] - peak, 0),
            "over_capacity": peak > mentor["capacity"],
        })

    reason_codes = defaultdict(int)
    reasons = {}
    for entry in plan["requested"]:
        if entry["recommendation"] != "pending":
            continue
        for code in entry["reason_codes"] or ["unspecified"]:
            reason_codes[code] += 1
        for reason in entry["reasons"] or ["未说明"]:
            reasons[reason] = reasons.get(reason, 0) + 1

    return {
        "vacancies": [per_vacancy[key] for key in sorted(per_vacancy)],
        "mentors": mentor_rows,
        "requested": plan["requested"],
        "retained": plan["retained"],
        "unassigned_reasons": dict(sorted(reasons.items())),
        "unassigned_reason_codes": dict(sorted(reason_codes.items())),
        "confirmed_registration_ids": sorted(
            registration["registration_id"]
            for registration in dataset.registrations.values()
            if registration["status"] == "confirmed"),
    }


__all__ = ["REASON_LABELS", "build_plan", "summarize_plan", "sessions_overlap",
           "mentor_of", "max_concurrent"]
