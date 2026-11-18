"""运营口径的分析视图。

把“原始平均分”“评分者尺度”“作品差异”和“调整后的学员能力”并排
显示，便于看出同一份评分记录在三种解释下会得出什么结论。
"""

from collections import defaultdict


def rater_scales(networks):
    """每个可比评分组内每位评分者的尺度（越大越严格）。"""
    rows = []
    for skill_id, network in sorted(networks.items()):
        for component in network.components:
            for rater_id, value in sorted(component["severity"].items()):
                rows.append({
                    "skill_id": skill_id,
                    "component_id": component["component_id"],
                    "rater_id": rater_id,
                    "severity": value,
                    "judgments": sum(1 for row in component["_judgments"]
                                     if row["rater_id"] == rater_id),
                })
    return rows


def raw_vs_adjusted(dataset, networks):
    """同一学员的原始平均分与调整后能力估计对照。"""
    rows = []
    for skill_id, network in sorted(networks.items()):
        for component in network.components:
            grouped = defaultdict(list)
            for judgment in component["_judgments"]:
                grouped[judgment["student_id"]].append(judgment["score"])
            for student_id, scores in sorted(grouped.items()):
                estimate = network.estimate(student_id)
                rows.append({
                    "student_id": student_id,
                    "skill_id": skill_id,
                    "component_id": component["component_id"],
                    "judgments": len(scores),
                    "works": len({row["work_id"] for row in component["_judgments"]
                                  if row["student_id"] == student_id}),
                    "raw_mean": sum(scores) / len(scores),
                    "adjusted": estimate["estimate"],
                    "interval": estimate["interval"],
                    "confidence": estimate["confidence"],
                    "evidence": estimate["evidence"],
                })
    return rows


def scale_decomposition(networks):
    """每个可比评分组里学员、作品、评分者、残余判断的变异份额。"""
    rows = []
    for skill_id, network in sorted(networks.items()):
        for component in network.components:
            rows.append({
                "skill_id": skill_id,
                "component_id": component["component_id"],
                "students": len(component["students"]),
                "raters": len(component["raters"]),
                "works": len(component["works"]),
                "judgments": component["judgment_count"],
                "converged": component["converged"],
                **component["decomposition"],
            })
    return rows


def connected_elsewhere(dataset, networks, student_id, skill_id):
    """说明某学员在该技能上是否根本不处在任何可比网络中。"""
    network = networks.get(skill_id)
    if network is None:
        return "该技能没有任何评分网络"
    component = network.locate(student_id)
    if component is None:
        return "该学员没有该技能的评分记录"
    others = [other for other in network.components if other is not component]
    return (f"属于 {component['component_id']}，与另外 {len(others)} 个"
            f"不连通评分组不能直接比较")


__all__ = ["rater_scales", "raw_vs_adjusted", "scale_decomposition",
           "connected_elsewhere"]
