"""可比能力网络。

评分是“某评分者在某技能上对某作品的一次判断”。直接取平均分会把评分者
尺度差异混进学员能力，因此这里把每条现行评分表示为：

    logit(y_ij) = 学员能力 θ_i − 评分者尺度 b_j + 误差

只在同一技能、且通过共同作品或共同评分者实际连接的评分组内估计。
互不连接的评分组各自定锚，绝不因为技能同名就排成统一名次。

拟合用交替最小二乘：先固定能力估计评分者尺度，再固定尺度估计能力，
直到变化小于容差；收敛与否如实报告。另给出每个评分组的方差分解，
供使用者区分学员能力差异、作品差异、评分者尺度差异和残余判断差异
各占多少。
"""

from collections import defaultdict
from math import exp, log

LOGIT_EPSILON = 1e-6


def to_logit(score, low, high):
    """把有界分数映射到 logit 尺度；端点做极小收缩以便计算。"""
    proportion = (score - low) / (high - low)
    proportion = min(max(proportion, LOGIT_EPSILON), 1 - LOGIT_EPSILON)
    return log(proportion / (1 - proportion))


def from_logit(value, low, high):
    proportion = 1 / (1 + exp(-value))
    return low + proportion * (high - low)


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def connected_components(judgments):
    """按技能分别求连通分量。

    作品和评分者、学员和作品之间建立连接；连接依据是共同作品或共同
    评分者，而不是技能名称。
    """
    forest = _UnionFind()
    for judgment in judgments:
        forest.union(("work", judgment["work_id"]), ("rater", judgment["rater_id"]))
        forest.union(("work", judgment["work_id"]), ("student", judgment["student_id"]))
    groups = defaultdict(list)
    for judgment in judgments:
        groups[(judgment["skill_id"], forest.find(("work", judgment["work_id"])))].append(judgment)
    by_skill = defaultdict(list)
    for (skill_id, _), members in sorted(groups.items()):
        by_skill[skill_id].append(members)
    return dict(by_skill)


def _solve_component(judgments, settings):
    """对单个连通评分组用交替最小二乘估计学员能力与评分者尺度。

    模型只有学员与评分者两个加性效应，因此可以在小样本上稳定求解；
    评分组内学员均值固定为 0，评分者尺度吸收整体宽松程度。不能收敛
    时保留最后一轮结果并如实标记。
    """
    students = sorted({row["student_id"] for row in judgments})
    raters = sorted({row["rater_id"] for row in judgments})
    works = sorted({row["work_id"] for row in judgments})
    values = {row["rating_id"]: row["logit"] for row in judgments}

    ability = {student: 0.0 for student in students}
    severity = {rater: 0.0 for rater in raters}

    converged = False
    used = 0
    for used in range(1, settings.max_iterations + 1):
        previous = (dict(ability), dict(severity))

        by_rater = defaultdict(list)
        for row in judgments:
            by_rater[row["rater_id"]].append(
                values[row["rating_id"]] - ability[row["student_id"]])
        for rater, parts in by_rater.items():
            severity[rater] = -sum(parts) / len(parts)

        by_student = defaultdict(list)
        for row in judgments:
            by_student[row["student_id"]].append(
                values[row["rating_id"]] + severity[row["rater_id"]])
        for student, parts in by_student.items():
            ability[student] = sum(parts) / len(parts)
        _center(ability, severity)

        change = max(
            [abs(ability[key] - previous[0][key]) for key in ability]
            + [abs(severity[key] - previous[1][key]) for key in severity]
        ) if judgments else 0.0
        if change <= settings.tolerance:
            converged = True
            break

    _center(ability, severity)
    residuals = {row["rating_id"]:
                 values[row["rating_id"]] - (ability[row["student_id"]]
                                             - severity[row["rater_id"]])
                 for row in judgments}
    return {
        "students": students,
        "raters": raters,
        "works": works,
        "ability": ability,
        "severity": severity,
        "residuals": residuals,
        "iterations": used,
        "converged": converged,
        "judgment_count": len(judgments),
    }


def _center(ability, severity):
    """固定评分组能力均值为 0，整体宽松程度进入评分者尺度。"""
    if not ability:
        return
    offset = sum(ability.values()) / len(ability)
    for key in ability:
        ability[key] -= offset
    for key in severity:
        severity[key] += offset


class SkillNetwork:
    """某技能下所有可比评分组的估计结果。"""

    def __init__(self, skill_id, components, settings):
        self.skill_id = skill_id
        self.settings = settings
        self.components = components

    @classmethod
    def fit(cls, skill_id, judgments, settings):
        members = [row for row in judgments if row["skill_id"] == skill_id]
        groups = connected_components(members).get(skill_id, [])
        # 按学员编号排序编号，保证同一档案反复计算的组名稳定。
        groups = sorted(groups, key=lambda group: sorted(
            {row["student_id"] for row in group}))
        components = []
        for index, group in enumerate(groups):
            component = _solve_component(group, settings)
            component["component_id"] = f"{skill_id}#{index + 1}"
            component["skill_id"] = skill_id
            component["_judgments"] = group
            component["decomposition"] = variance_decomposition(component)
            components.append(component)
        return cls(skill_id, components, settings)

    def locate(self, student_id):
        for component in self.components:
            if student_id in component["ability"]:
                return component
        return None

    def estimate(self, student_id):
        """返回学员能力估计、标准误、置信区间、依据评分和证据充分性。"""
        component = self.locate(student_id)
        if component is None:
            return {
                "skill_id": self.skill_id,
                "component_id": None,
                "estimate": None,
                "standard_error": None,
                "interval": None,
                "rating_ids": [],
                "work_ids": [],
                "rater_ids": [],
                "evidence": "缺少可用评分",
                "confidence": "none",
                "evidence_code": "no_judgments",
            }
        ratings = [row for row in component["_judgments"]
                   if row["student_id"] == student_id]
        estimate = component["ability"][student_id]
        standard_error = _standard_error(component, student_id, self.settings)
        z = self.settings.z
        evidence, confidence, evidence_code = self._evidence(component, ratings)
        return {
            "skill_id": self.skill_id,
            "component_id": component["component_id"],
            "estimate": estimate,
            "standard_error": standard_error,
            "interval": (estimate - z * standard_error,
                         estimate + z * standard_error),
            "rating_ids": sorted(row["rating_id"] for row in ratings),
            "work_ids": sorted({row["work_id"] for row in ratings}),
            "rater_ids": sorted({row["rater_id"] for row in ratings}),
            "evidence": evidence,
            "confidence": confidence,
            "evidence_code": evidence_code,
        }

    def _evidence(self, component, ratings):
        """返回人类可读说明与机器可读的证据码。"""
        settings = self.settings
        works = {row["work_id"] for row in ratings}
        if len(ratings) < settings.min_judgments or len(works) < settings.min_works:
            return (f"证据不足：{len(ratings)} 条评分 / {len(works)} 件作品，"
                    f"未达到 {settings.min_judgments} 条且 {settings.min_works} 件",
                    "insufficient", "insufficient_judgments")
        if len(component["raters"]) < settings.min_component_raters:
            return ("尺度不可分离：评分组只有一位评分者，"
                    "无法区分学员能力与评分尺度",
                    "insufficient", "scale_inseparable")
        if len(component["students"]) < settings.min_component_students:
            return ("尺度不可分离：评分组只有一位学员，"
                    "无法区分学员能力与评分尺度",
                    "insufficient", "scale_inseparable")
        if not component["converged"]:
            return ("估计未收敛，不能用于资格判断", "insufficient", "not_converged")
        return (f"可比评分组 {component['component_id']} 内 "
                f"{len(component['raters'])} 位评分者、"
                f"{len(component['students'])} 位学员", "usable", None)

    def required_level(self, student_id, threshold):
        """要求水平是否落在能力置信区间之上、之下或之内。

        置信区间与阈值刚好相接（端点相等）按“不确定”处理，避免把
        数值上的巧合当成资格证据。
        """
        result = self.estimate(student_id)
        if result["estimate"] is None or result["confidence"] != "usable":
            return "uncertain", result
        low, high = result["interval"]
        if low > threshold:
            return "met", result
        if high < threshold:
            return "not_met", result
        return "uncertain", result

    def rank_within_component(self):
        """仅在单个可比评分组内排序；不跨组拼接。"""
        result = {}
        for component in self.components:
            ordered = sorted(component["students"],
                             key=lambda student: component["ability"][student],
                             reverse=True)
            for position, student in enumerate(ordered, start=1):
                result[student] = {"component_id": component["component_id"],
                                   "position": position,
                                   "of": len(ordered)}
        return result


def _standard_error(component, student_id, settings):
    """组内残差与评分条数给出的近似标准误。

    残差方差在观测值与先验之间做收缩，避免 1–2 条评分时把标准误
    压到 0、给人以“区间很窄”的错觉。
    """
    squared = [value * value for value in component["residuals"].values()]
    prior = settings.residual_prior_sd ** 2
    weight = settings.residual_prior_weight
    residual_variance = ((sum(squared) + weight * prior)
                         / (len(squared) + weight + 1))
    count = sum(1 for row in component["_judgments"]
                if row["student_id"] == student_id)
    return ((residual_variance / max(count, 1)) ** 0.5) * (1 + 1 / max(count, 1))


def _weighted_variance(judgments, function):
    values = [function(row) for row in judgments]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def variance_decomposition(component):
    """按估计模型分解评分变异。

    学员能力、评分者尺度、作品平均残差和组内残余判断各得到一份
    平方和，份额按四者之和归一化，因此总和为 1。作品只有一位评分者
    时无法与判断噪声区分，作品份额会如实变小而不是被高估。
    """
    judgments = component["_judgments"]
    if not judgments:
        return {"student": 0.0, "work": 0.0, "rater": 0.0, "residual": 0.0}
    work_means = {}
    for work, rows in _grouped(judgments, "work_id").items():
        work_means[work] = sum(component["residuals"][row["rating_id"]]
                               for row in rows) / len(rows)

    parts = {
        "student": _weighted_variance(
            judgments, lambda row: component["ability"][row["student_id"]]),
        "rater": _weighted_variance(
            judgments, lambda row: component["severity"][row["rater_id"]]),
        "work": _weighted_variance(
            judgments, lambda row: work_means[row["work_id"]]),
        "residual": _weighted_variance(
            judgments,
            lambda row: component["residuals"][row["rating_id"]]
            - work_means[row["work_id"]]),
    }
    total = sum(parts.values())
    if total == 0:
        return {key: 0.0 for key in parts}
    return {key: value / total for key, value in parts.items()}


def _grouped(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def build_networks(dataset):
    """构造所有技能的现行评分网络，并记录每条评分替换掉的旧评分。"""
    superseded = dataset.superseded_by
    by_skill = defaultdict(list)
    for rating in dataset.current_ratings():
        by_skill[rating["skill_id"]].append({
            "rating_id": rating["rating_id"],
            "work_id": rating["work_id"],
            "student_id": rating["student_id"],
            "skill_id": rating["skill_id"],
            "rater_id": rating["rater_id"],
            "rated_at": rating["rated_at"],
            "score": rating["score"],
            "logit": to_logit(rating["score"],
                              dataset.settings.score_min, dataset.settings.score_max),
            "superseded_rating_ids": superseded_chain(rating["rating_id"], superseded),
        })
    return {skill_id: SkillNetwork.fit(skill_id, members, dataset.settings)
            for skill_id, members in by_skill.items()}


def superseded_chain(rating_id, superseded):
    """返回被该现行评分直接或间接替换掉的旧评分编号。"""
    replaced = defaultdict(list)
    for old, new in superseded.items():
        replaced[new].append(old)
    result = []
    stack = list(replaced.get(rating_id, []))
    while stack:
        current = stack.pop()
        result.append(current)
        stack.extend(replaced.get(current, []))
    return sorted(result)


def coverage(networks):
    """报告每个技能的可比评分网络覆盖到了哪里。"""
    result = []
    for skill_id, network in sorted(networks.items()):
        total_students = {student for component in network.components
                          for student in component["students"]}
        result.append({
            "skill_id": skill_id,
            "students": len(total_students),
            "components": [{
                "component_id": component["component_id"],
                "students": len(component["students"]),
                "raters": len(component["raters"]),
                "works": len(component["works"]),
                "judgments": component["judgment_count"],
                "converged": component["converged"],
                "decomposition": component["decomposition"],
            } for component in network.components],
        })
    return result
