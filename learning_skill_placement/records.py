"""读取并校验档案记录。

档案由若干 CSV 组成：技能、学员、作品、评分、岗位、服务时段、
带教者、已确定报名和当时的资格决定。读取层只做结构校验和引用
完整性检查，不解释评分尺度。
"""

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .settings import Settings

TABLE_FIELDS = {
    "skills": {"skill_id", "name"},
    "students": {"student_id", "name", "community"},
    "works": {"work_id", "student_id", "title", "submitted_at"},
    "ratings": {"rating_id", "work_id", "student_id", "skill_id", "rater_id",
                "score", "rated_at", "replaces_rating_id"},
    "vacancies": {"vacancy_id", "title", "site", "required_skills"},
    "sessions": {"session_id", "vacancy_id", "date", "start", "end",
                 "independent_slots", "supervised_slots", "mentor_id"},
    "mentors": {"mentor_id", "name", "capacity"},
    "registrations": {"registration_id", "vacancy_id", "session_id", "student_id",
                      "mode", "mentor_id", "status", "decision_id"},
    "decisions": {"decision_id", "student_id", "vacancy_id", "decided_at",
                  "outcome", "reason", "evidence_rating_ids", "estimates_json"},
}

DECISION_OUTCOMES = {"confirmed", "pending_evidence", "declined"}


def parse_requirements(text):
    """把 ``S-COM:-0.1;S-ORG:0.2`` 解析为技能与要求水平。"""
    result = []
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"必要技能格式应为 技能:水平，得到 {part!r}")
        skill_id, level = part.rsplit(":", 1)
        try:
            level = float(level)
        except ValueError as error:
            raise ValueError(f"技能要求水平不是数字: {part!r}") from error
        result.append({"skill_id": skill_id.strip(), "level": level})
    if not result:
        raise ValueError("岗位必须至少有一项必要技能")
    return result


def _read_csv(path, required):
    if not path.exists():
        raise FileNotFoundError(f"缺少档案文件: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path.name} 缺少字段: {', '.join(sorted(missing))}")
        rows = []
        for raw in reader:
            row = {key: (value.strip() if isinstance(value, str) else value)
                   for key, value in raw.items() if key is not None}
            if not any(row.values()):
                continue
            rows.append(row)
    return rows


def _unique(rows, key, table):
    seen = set()
    for row in rows:
        value = row[key]
        if not value:
            raise ValueError(f"{table} 中存在空的 {key}")
        if value in seen:
            raise ValueError(f"{table} 中 {key} 重复: {value}")
        seen.add(value)
    return {row[key]: row for row in rows}


def _integer(value, field, table):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{table} 的 {field} 必须是整数: {value!r}") from error
    if number < 0:
        raise ValueError(f"{table} 的 {field} 不能为负数: {number}")
    return number


def _number(value, field, table):
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{table} 的 {field} 必须是数字: {value!r}") from error


def _check_date(value, field, table):
    try:
        date.fromisoformat(str(value).split("T", 1)[0])
    except ValueError as error:
        raise ValueError(f"{table} 的 {field} 不是有效日期: {value!r}") from error
    return str(value)


@dataclass
class Dataset:
    skills: dict
    students: dict
    works: dict
    ratings: dict
    vacancies: dict
    sessions: dict
    mentors: dict
    registrations: dict
    decisions: dict
    settings: Settings

    @property
    def current_rating_ids(self):
        replaced = {row["replaces_rating_id"] for row in self.ratings.values()
                    if row.get("replaces_rating_id")}
        return {rating_id for rating_id in self.ratings if rating_id not in replaced}

    @property
    def superseded_by(self):
        mapping = {}
        for row in self.ratings.values():
            previous = row.get("replaces_rating_id")
            if previous:
                mapping[previous] = row["rating_id"]
        return mapping

    def current_ratings(self):
        current = self.current_rating_ids
        return [row for rating_id, row in self.ratings.items() if rating_id in current]

    def rating_chain(self, rating_id):
        """返回从该评分起逐次替换到最新评分的编号链。"""
        chain = [rating_id]
        replacements = self.superseded_by
        while chain[-1] in replacements:
            chain.append(replacements[chain[-1]])
        return chain


def load_dataset(root, settings=None):
    settings = settings or Settings()
    root = Path(root)
    raw = {table: _read_csv(root / f"{table}.csv", fields)
           for table, fields in TABLE_FIELDS.items()}
    return build_dataset(raw, settings)


def build_dataset(raw, settings=None):
    """从已读取的各表记录构造并校验数据集。"""
    settings = settings or Settings()
    missing_tables = set(TABLE_FIELDS) - set(raw)
    if missing_tables:
        raise ValueError(f"缺少档案表: {', '.join(sorted(missing_tables))}")
    for table, fields in TABLE_FIELDS.items():
        for row in raw[table]:
            missing = fields - set(row)
            if missing:
                raise ValueError(f"{table} 缺少字段: {', '.join(sorted(missing))}")

    skills = _unique(raw["skills"], "skill_id", "skills")
    students = _unique(raw["students"], "student_id", "students")
    works = _unique(raw["works"], "work_id", "works")
    ratings = _unique(raw["ratings"], "rating_id", "ratings")
    vacancies = _unique(raw["vacancies"], "vacancy_id", "vacancies")
    sessions = _unique(raw["sessions"], "session_id", "sessions")
    mentors = _unique(raw["mentors"], "mentor_id", "mentors")
    registrations = _unique(raw["registrations"], "registration_id", "registrations")
    decisions = _unique(raw["decisions"], "decision_id", "decisions")

    for work in works.values():
        if work["student_id"] not in students:
            raise ValueError(f"作品 {work['work_id']} 指向未知学员 {work['student_id']}")
        _check_date(work["submitted_at"], "submitted_at", "works")

    replaced_by = {}
    for rating in ratings.values():
        work = works.get(rating["work_id"])
        if work is None:
            raise ValueError(f"评分 {rating['rating_id']} 指向未知作品 {rating['work_id']}")
        if rating["student_id"] not in students:
            raise ValueError(f"评分 {rating['rating_id']} 指向未知学员 {rating['student_id']}")
        if rating["student_id"] != work["student_id"]:
            raise ValueError(f"评分 {rating['rating_id']} 的学员与作品作者不一致")
        if rating["skill_id"] not in skills:
            raise ValueError(f"评分 {rating['rating_id']} 指向未知技能 {rating['skill_id']}")
        if not rating["rater_id"]:
            raise ValueError(f"评分 {rating['rating_id']} 缺少评分者")
        rating["score"] = _number(rating["score"], "score", "ratings")
        if not settings.score_min <= rating["score"] <= settings.score_max:
            raise ValueError(
                f"评分 {rating['rating_id']} 的分数 {rating['score']} 超出 "
                f"{settings.score_min:g}–{settings.score_max:g}")
        _check_date(rating["rated_at"], "rated_at", "ratings")
        previous = rating.get("replaces_rating_id") or None
        rating["replaces_rating_id"] = previous
        if previous:
            if previous not in ratings:
                raise ValueError(f"评分 {rating['rating_id']} 替换了未知评分 {previous}")
            if previous in replaced_by:
                raise ValueError(f"评分 {previous} 被多条重评替换")
            old = ratings[previous]
            if old["work_id"] != rating["work_id"] or old["skill_id"] != rating["skill_id"]:
                raise ValueError(f"重评 {rating['rating_id']} 必须针对同一作品与技能")
            if old["rated_at"] > rating["rated_at"]:
                raise ValueError(f"重评 {rating['rating_id']} 的时间早于被替换评分")
            replaced_by[previous] = rating["rating_id"]

    current = {rating_id for rating_id in ratings if rating_id not in replaced_by}
    seen_judgment = set()
    for rating_id in sorted(current):
        rating = ratings[rating_id]
        key = (rating["work_id"], rating["skill_id"], rating["rater_id"])
        if key in seen_judgment:
            raise ValueError(
                f"作品 {rating['work_id']} 的技能 {rating['skill_id']} 存在同一评分者的"
                f"多条现行评分，请用 replaces_rating_id 标明重评关系")
        seen_judgment.add(key)

    for rating_id in ratings:
        walk, seen = rating_id, set()
        while walk:
            if walk in seen:
                raise ValueError(f"重评链在 {rating_id} 处形成循环")
            seen.add(walk)
            walk = ratings[walk]["replaces_rating_id"]

    for vacancy in vacancies.values():
        vacancy["requirements"] = parse_requirements(vacancy["required_skills"])
        for requirement in vacancy["requirements"]:
            if requirement["skill_id"] not in skills:
                raise ValueError(
                    f"岗位 {vacancy['vacancy_id']} 要求未知技能 {requirement['skill_id']}")

    for session in sessions.values():
        if session["vacancy_id"] not in vacancies:
            raise ValueError(f"时段 {session['session_id']} 指向未知岗位 {session['vacancy_id']}")
        _check_date(session["date"], "date", "sessions")
        for field in ("start", "end"):
            value = session[field]
            try:
                hours, minutes = (int(part) for part in value.split(":"))
                if not (0 <= hours < 24 and 0 <= minutes < 60):
                    raise ValueError
            except ValueError as error:
                raise ValueError(
                    f"时段 {session['session_id']} 的 {field} 不是有效时间: {value!r}") from error
        if minutes_of(session["end"]) <= minutes_of(session["start"]):
            raise ValueError(f"时段 {session['session_id']} 的结束时间必须晚于开始时间")
        # 统一补零，保证字符串比较等同于时间先后比较。
        session["start"], session["end"] = _clock(session["start"]), _clock(session["end"])
        session["independent_slots"] = _integer(
            session["independent_slots"], "independent_slots", "sessions")
        session["supervised_slots"] = _integer(
            session["supervised_slots"], "supervised_slots", "sessions")
        if session["mentor_id"] and session["mentor_id"] not in mentors:
            raise ValueError(
                f"时段 {session['session_id']} 指向未知带教者 {session['mentor_id']}")

    for mentor in mentors.values():
        mentor["capacity"] = _integer(mentor["capacity"], "capacity", "mentors")
        if mentor["capacity"] < 1:
            raise ValueError(f"带教者 {mentor['mentor_id']} 的同时带教容量至少为 1")

    for decision in decisions.values():
        if decision["student_id"] not in students:
            raise ValueError(f"决定 {decision['decision_id']} 指向未知学员")
        if decision["vacancy_id"] not in vacancies:
            raise ValueError(f"决定 {decision['decision_id']} 指向未知岗位")
        _check_date(decision["decided_at"], "decided_at", "decisions")
        if decision["outcome"] not in DECISION_OUTCOMES:
            raise ValueError(
                f"决定 {decision['decision_id']} 的 outcome 必须是 "
                f"{' / '.join(sorted(DECISION_OUTCOMES))}")
        evidence = decision["evidence_rating_ids"]
        if isinstance(evidence, str):
            evidence = [part.strip() for part in evidence.split(";") if part.strip()]
        else:
            evidence = [str(part).strip() for part in evidence if str(part).strip()]
        for rating_id in evidence:
            if rating_id not in ratings:
                raise ValueError(
                    f"决定 {decision['decision_id']} 引用了不存在于档案中的评分 {rating_id}")
        decision["evidence_rating_ids"] = evidence
        try:
            decision["estimates"] = json.loads(decision["estimates_json"] or "{}")
        except json.JSONDecodeError as error:
            raise ValueError(
                f"决定 {decision['decision_id']} 的 estimates_json 不是有效 JSON") from error

    confirmed_slots = set()
    for registration in registrations.values():
        if registration["student_id"] not in students:
            raise ValueError(f"报名 {registration['registration_id']} 指向未知学员")
        if registration["vacancy_id"] not in vacancies:
            raise ValueError(f"报名 {registration['registration_id']} 指向未知岗位")
        if registration["session_id"] not in sessions:
            raise ValueError(f"报名 {registration['registration_id']} 指向未知时段")
        if sessions[registration["session_id"]]["vacancy_id"] != registration["vacancy_id"]:
            raise ValueError(f"报名 {registration['registration_id']} 的岗位与时段不一致")
        if registration["mode"] not in {"independent", "supervised"}:
            raise ValueError(
                f"报名 {registration['registration_id']} 的 mode 必须是 independent 或 supervised")
        if registration["status"] not in {"requested", "confirmed", "withdrawn"}:
            raise ValueError(
                f"报名 {registration['registration_id']} 的 status 必须是 "
                "requested、confirmed 或 withdrawn")
        if registration["status"] == "requested" and registration["decision_id"]:
            raise ValueError(
                f"待审核报名 {registration['registration_id']} 不应预先引用决定")
        if registration["mode"] == "supervised" and registration["status"] != "withdrawn" \
                and not (registration["mentor_id"]
                         or sessions[registration["session_id"]]["mentor_id"]):
            raise ValueError(
                f"带教报名 {registration['registration_id']} 没有可用的带教者")
        if registration["mentor_id"] and registration["mentor_id"] not in mentors:
            raise ValueError(f"报名 {registration['registration_id']} 指向未知带教者")
        if registration["status"] == "confirmed":
            decision = decisions.get(registration["decision_id"])
            if decision is None:
                raise ValueError(
                    f"已确定报名 {registration['registration_id']} 缺少对应的决定记录")
            if decision["student_id"] != registration["student_id"] or \
                    decision["vacancy_id"] != registration["vacancy_id"]:
                raise ValueError(
                    f"报名 {registration['registration_id']} 的决定与学员或岗位不一致")
            if decision["outcome"] not in {"confirmed", "pending_evidence"}:
                raise ValueError(
                    f"已确定报名 {registration['registration_id']} 引用了一个未确认的决定")
            if registration["mode"] == "supervised" and not registration["mentor_id"]:
                raise ValueError(
                    f"带教报名 {registration['registration_id']} 必须指定带教者")
            slot = (registration["session_id"], registration["student_id"])
            if slot in confirmed_slots:
                raise ValueError(
                    f"学员 {registration['student_id']} 在时段 "
                    f"{registration['session_id']} 有重复的已确定报名")
            confirmed_slots.add(slot)

    return Dataset(skills, students, works, ratings, vacancies, sessions,
                   mentors, registrations, decisions, settings)


def minutes_of(value):
    hours, minutes = (int(part) for part in value.split(":"))
    return hours * 60 + minutes


def _clock(value):
    """把 9:00 之类写法规范成 09:00，供字符串比较使用。"""
    hours, minutes = (int(part) for part in value.split(":"))
    return f"{hours:02d}:{minutes:02d}"
