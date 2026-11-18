"""把档案写入 SQLite 并保留可追溯的评分链。

数据库只是本地核对用的派生物：每次运行从 CSV 重建，原始档案不被
改写。写入时保留重评关系，使“当时依据”和“现行证据”可以分别查询。
"""

import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY, name TEXT NOT NULL, community TEXT);
CREATE TABLE IF NOT EXISTS works (
    work_id TEXT PRIMARY KEY, student_id TEXT NOT NULL, title TEXT,
    submitted_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ratings (
    rating_id TEXT PRIMARY KEY, work_id TEXT NOT NULL, student_id TEXT NOT NULL,
    skill_id TEXT NOT NULL, rater_id TEXT NOT NULL, score REAL NOT NULL,
    rated_at TEXT NOT NULL, replaces_rating_id TEXT,
    FOREIGN KEY (replaces_rating_id) REFERENCES ratings(rating_id));
CREATE TABLE IF NOT EXISTS vacancies (
    vacancy_id TEXT PRIMARY KEY, title TEXT, site TEXT,
    required_skills TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, vacancy_id TEXT NOT NULL, date TEXT NOT NULL,
    start TEXT NOT NULL, end TEXT NOT NULL, independent_slots INTEGER NOT NULL,
    supervised_slots INTEGER NOT NULL, mentor_id TEXT);
CREATE TABLE IF NOT EXISTS mentors (
    mentor_id TEXT PRIMARY KEY, name TEXT, capacity INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY, student_id TEXT NOT NULL,
    vacancy_id TEXT NOT NULL, decided_at TEXT NOT NULL, outcome TEXT NOT NULL,
    reason TEXT, evidence_rating_ids TEXT, estimates_json TEXT);
CREATE TABLE IF NOT EXISTS registrations (
    registration_id TEXT PRIMARY KEY, vacancy_id TEXT NOT NULL,
    session_id TEXT NOT NULL, student_id TEXT NOT NULL, mode TEXT NOT NULL,
    mentor_id TEXT, status TEXT NOT NULL, decision_id TEXT);
CREATE INDEX IF NOT EXISTS ratings_by_skill ON ratings(skill_id);
CREATE INDEX IF NOT EXISTS ratings_by_rater ON ratings(rater_id);
"""

TABLES = ["skills", "students", "works", "ratings", "vacancies", "sessions",
          "mentors", "decisions", "registrations"]

COLUMNS = {
    "skills": ["skill_id", "name"],
    "students": ["student_id", "name", "community"],
    "works": ["work_id", "student_id", "title", "submitted_at"],
    "ratings": ["rating_id", "work_id", "student_id", "skill_id", "rater_id",
                "score", "rated_at", "replaces_rating_id"],
    "vacancies": ["vacancy_id", "title", "site", "required_skills"],
    "sessions": ["session_id", "vacancy_id", "date", "start", "end",
                 "independent_slots", "supervised_slots", "mentor_id"],
    "mentors": ["mentor_id", "name", "capacity"],
    "decisions": ["decision_id", "student_id", "vacancy_id", "decided_at",
                  "outcome", "reason", "evidence_rating_ids", "estimates_json"],
    "registrations": ["registration_id", "vacancy_id", "session_id", "student_id",
                      "mode", "mentor_id", "status", "decision_id"],
}


def _value(row, column):
    value = row.get(column)
    if column == "evidence_rating_ids" and isinstance(value, list):
        return ";".join(value)
    if column == "estimates" and isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if column == "estimates_json":
        if "estimates" in row:
            return json.dumps(row["estimates"], ensure_ascii=False, sort_keys=True)
        return value
    return value


def write_dataset(dataset, path):
    """把数据集写入全新的 SQLite 文件，返回写入行数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        connection.execute("BEGIN")
        # 重评记录可能先于其引用的旧评分出现，事务结束时再检查外键。
        connection.execute("PRAGMA defer_foreign_keys = ON")
        for table in reversed(TABLES):
            connection.execute(f"DELETE FROM {table}")
        counts = {}
        for table in TABLES:
            rows = getattr(dataset, table).values()
            columns = COLUMNS[table]
            placeholders = ", ".join("?" for _ in columns)
            payload = [tuple(_value(row, column) for column in columns)
                       for row in rows]
            connection.executemany(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                payload)
            counts[table] = len(payload)
        connection.commit()
    return counts


def current_ratings(path, skill_id=None):
    """从数据库读回现行评分（未被任何重评替换的评分）。"""
    query = ("SELECT r.rating_id, r.work_id, r.student_id, r.skill_id, r.rater_id, "
             "r.score, r.rated_at FROM ratings r WHERE NOT EXISTS "
             "(SELECT 1 FROM ratings n WHERE n.replaces_rating_id = r.rating_id)")
    parameters = ()
    if skill_id:
        query += " AND r.skill_id = ?"
        parameters = (skill_id,)
    query += " ORDER BY r.rating_id"
    with sqlite3.connect(path) as connection:
        return connection.execute(query, parameters).fetchall()


def read_dataset(root):
    """从目录读取数据集（见 records.load_dataset），便于报告与测试共用。"""
    from .records import load_dataset
    return load_dataset(root)
