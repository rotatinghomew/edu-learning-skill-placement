import tempfile
import unittest
from pathlib import Path

from learning_skill_placement.analysis import (raw_vs_adjusted, rater_scales,
                                               scale_decomposition)
from learning_skill_placement.network import build_networks, coverage, from_logit
from learning_skill_placement.placement import summarize_plan
from learning_skill_placement.qualification import (confirmed_standing,
                                                    qualification_evidence,
                                                    trace_decision)
from learning_skill_placement.records import build_dataset
from learning_skill_placement.report import render
from learning_skill_placement.storage import current_ratings, write_dataset


def skill(skill_id, name="技能"):
    return {"skill_id": skill_id, "name": name}


def student(student_id, community="社区甲"):
    return {"student_id": student_id, "name": student_id, "community": community}


def work(work_id, student_id, submitted_at="2026-09-01"):
    return {"work_id": work_id, "student_id": student_id,
            "title": work_id, "submitted_at": submitted_at}


def rating(rating_id, work_id, student_id, skill_id, rater_id, score,
           rated_at="2026-09-05", replaces_rating_id=""):
    return {"rating_id": rating_id, "work_id": work_id,
            "student_id": student_id, "skill_id": skill_id,
            "rater_id": rater_id, "score": score, "rated_at": rated_at,
            "replaces_rating_id": replaces_rating_id}


def vacancy(vacancy_id, required_skills):
    return {"vacancy_id": vacancy_id, "title": vacancy_id, "site": "站点",
            "required_skills": required_skills}


def session(session_id, vacancy_id, date="2026-12-05", start="09:00",
            end="11:00", independent_slots=1, supervised_slots=1,
            mentor_id=""):
    return {"session_id": session_id, "vacancy_id": vacancy_id,
            "date": date, "start": start, "end": end,
            "independent_slots": independent_slots,
            "supervised_slots": supervised_slots, "mentor_id": mentor_id}


def mentor(mentor_id, capacity=1):
    return {"mentor_id": mentor_id, "name": mentor_id, "capacity": capacity}


def registration(registration_id, vacancy_id, session_id, student_id,
                 mode="independent", mentor_id="", status="requested",
                 decision_id=""):
    return {"registration_id": registration_id, "vacancy_id": vacancy_id,
            "session_id": session_id, "student_id": student_id, "mode": mode,
            "mentor_id": mentor_id, "status": status,
            "decision_id": decision_id}


def decision(decision_id, student_id, vacancy_id, evidence_rating_ids,
             outcome="confirmed", decided_at="2026-09-10", reason="当时依据",
             estimates_json="{}"):
    return {"decision_id": decision_id, "student_id": student_id,
            "vacancy_id": vacancy_id, "decided_at": decided_at,
            "outcome": outcome, "reason": reason,
            "evidence_rating_ids": evidence_rating_ids,
            "estimates_json": estimates_json}


def make(settings=None, skills=None, students=None, works=None, ratings=None,
         vacancies=None, sessions=None, mentors=None, registrations=None,
         decisions=None):
    raw = {
        "skills": skills or [],
        "students": students or [],
        "works": works or [],
        "ratings": ratings or [],
        "vacancies": vacancies or [],
        "sessions": sessions or [],
        "mentors": mentors or [],
        "registrations": registrations or [],
        "decisions": decisions or [],
    }
    return build_dataset(raw, settings)


def scale_archive(raters=("T1", "T2"), abilities=None):
    """两名评分者交叉评两位学员，用于检验尺度扣除。"""
    abilities = abilities or {"a": 0.0, "b": 0.0}
    ratings, works, students = [], [], []
    for student_id, ability in abilities.items():
        students.append(student(student_id))
        for index in (1, 2):
            work_id = f"W{student_id}{index}"
            works.append(work(work_id, student_id))
            for rater_id, severity in zip(raters, (0.5, -0.5)):
                score = round(from_logit(ability - severity, 0.0, 100.0), 2)
                ratings.append(rating(
                    f"R{student_id}{index}{rater_id}", work_id, student_id,
                    "S", rater_id, score))
    return make(skills=[skill("S")], students=students, works=works,
                ratings=ratings)


class RecordReadingTests(unittest.TestCase):
    def test_sample_archive_loads_and_has_expected_current_ratings(self):
        path = Path(__file__).parent.parent / "data"
        from learning_skill_placement.records import load_dataset
        dataset = load_dataset(path)
        self.assertEqual(len(dataset.current_ratings()), 61)
        self.assertEqual(dataset.ratings["R26"]["replaces_rating_id"], "R01")
        self.assertNotIn("R01", dataset.current_rating_ids)
        self.assertEqual(dataset.rating_chain("R01"), ["R01", "R26"])

    def test_missing_business_field_is_rejected(self):
        from learning_skill_placement.records import TABLE_FIELDS
        raw = {table: [] for table in TABLE_FIELDS}
        raw["ratings"] = [{"rating_id": "R1"}]
        with self.assertRaisesRegex(ValueError, "score"):
            build_dataset(raw)

    def test_revision_must_target_same_work_and_skill(self):
        dataset_rows = dict(
            skills=[skill("S"), skill("T")],
            students=[student("a")],
            works=[work("W1", "a"), work("W2", "a")],
            ratings=[
                rating("R1", "W1", "a", "S", "T1", 60),
                rating("R2", "W2", "a", "S", "T1", 70, "2026-09-06", "R1"),
            ])
        with self.assertRaisesRegex(ValueError, "同一作品与技能"):
            make(**dataset_rows)

    def test_duplicate_current_judgment_must_be_marked_as_revision(self):
        with self.assertRaisesRegex(ValueError, "多条现行评分"):
            make(skills=[skill("S")], students=[student("a")],
                 works=[work("W1", "a")],
                 ratings=[rating("R1", "W1", "a", "S", "T1", 60),
                          rating("R2", "W1", "a", "S", "T1", 50)])

    def test_scores_outside_range_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "超出"):
            make(skills=[skill("S")], students=[student("a")],
                 works=[work("W1", "a")],
                 ratings=[rating("R1", "W1", "a", "S", "T1", 120)])

    def test_confirmed_registration_requires_a_decision(self):
        with self.assertRaisesRegex(ValueError, "缺少对应的决定记录"):
            make(skills=[skill("S")], students=[student("a")],
                 works=[work("W1", "a")],
                 vacancies=[vacancy("V", "S:0")],
                 sessions=[session("SE", "V", independent_slots=1,
                                   supervised_slots=0)],
                 registrations=[registration("G1", "V", "SE", "a",
                                             status="confirmed")])


class NetworkTests(unittest.TestCase):
    def test_rater_scale_does_not_become_student_ability(self):
        # a 与 b 真实能力相同：a 同时被宽松与严格评分者评，b 只被宽松
        # 评分者评。原始均分 b 明显更高，调整后两人应不可区分。
        ratings, works = [], []
        for student_id in ("a", "b"):
            works.append(work(f"W{student_id}1", student_id))
            works.append(work(f"W{student_id}2", student_id))
        harsh, lenient = 0.5, -0.5
        ratings += [
            rating("R1", "Wa1", "a", "S", "T1", round(from_logit(-harsh, 0, 100), 2)),
            rating("R2", "Wa1", "a", "S", "T2", round(from_logit(-lenient, 0, 100), 2)),
            rating("R3", "Wa2", "a", "S", "T1", round(from_logit(-harsh, 0, 100), 2)),
            rating("R4", "Wa2", "a", "S", "T2", round(from_logit(-lenient, 0, 100), 2)),
            rating("R5", "Wb1", "b", "S", "T2", round(from_logit(-lenient, 0, 100), 2)),
            rating("R6", "Wb2", "b", "S", "T2", round(from_logit(-lenient, 0, 100), 2)),
        ]
        dataset = make(skills=[skill("S")], students=[student("a"), student("b")],
                       works=works, ratings=ratings)
        network = build_networks(dataset)["S"]
        component = network.components[0]
        self.assertGreater(component["severity"]["T1"],
                           component["severity"]["T2"])
        estimates = {row["student_id"]: row for row in
                     raw_vs_adjusted(dataset, {"S": network})}
        self.assertGreater(estimates["b"]["raw_mean"],
                           estimates["a"]["raw_mean"])
        self.assertAlmostEqual(estimates["a"]["adjusted"], 0.0, places=2)
        self.assertAlmostEqual(estimates["b"]["adjusted"], 0.0, places=2)

    def test_separate_components_are_not_ranked_together(self):
        # 四个学员两两成组：a/b 由 T1 连接，c/d 由 T2 连接。
        dataset = make(
            skills=[skill("S")],
            students=[student(x) for x in "abcd"],
            works=[work("W1", "a"), work("W2", "b"),
                   work("W3", "c"), work("W4", "d")],
            ratings=[rating("R1", "W1", "a", "S", "T1", 70),
                     rating("R2", "W2", "b", "S", "T1", 60),
                     rating("R3", "W3", "c", "S", "T2", 80),
                     rating("R4", "W4", "d", "S", "T2", 50)])
        network = build_networks(dataset)["S"]
        self.assertEqual(len(network.components), 2)
        ranks = network.rank_within_component()
        self.assertEqual(ranks["a"]["component_id"],
                         ranks["b"]["component_id"])
        self.assertNotEqual(ranks["a"]["component_id"],
                            ranks["c"]["component_id"])
        self.assertEqual(ranks["a"]["of"], 2)
        self.assertEqual(ranks["c"]["of"], 2)
        self.assertEqual(network.estimate("a")["component_id"],
                         ranks["a"]["component_id"])
        self.assertEqual(len(coverage({"S": network})[0]["components"]), 2)

    def test_single_rater_component_cannot_separate_ability_from_scale(self):
        dataset = make(
            skills=[skill("S")],
            students=[student("a"), student("b")],
            works=[work("W1", "a"), work("W2", "a"),
                   work("W3", "b"), work("W4", "b")],
            ratings=[rating("R1", "W1", "a", "S", "T9", 80),
                     rating("R2", "W2", "a", "S", "T9", 82),
                     rating("R3", "W3", "b", "S", "T9", 50),
                     rating("R4", "W4", "b", "S", "T9", 52)])
        network = build_networks(dataset)["S"]
        estimate = network.estimate("a")
        self.assertEqual(estimate["confidence"], "insufficient")
        self.assertIn("尺度不可分离", estimate["evidence"])
        status, _ = network.required_level("a", 0.0)
        self.assertEqual(status, "uncertain")

    def test_thin_evidence_is_uncertain_not_default_pass_or_fail(self):
        dataset = make(
            skills=[skill("S")],
            students=[student("a")],
            works=[work("W1", "a")],
            ratings=[rating("R1", "W1", "a", "S", "T1", 90)])
        estimate = build_networks(dataset)["S"].estimate("a")
        self.assertEqual(estimate["confidence"], "insufficient")
        self.assertIn("证据不足", estimate["evidence"])

    def test_decomposition_reports_all_four_sources(self):
        dataset = scale_archive(abilities={"a": 0.5, "b": -0.5})
        rows = scale_decomposition(build_networks(dataset))
        self.assertEqual(len(rows), 1)
        parts = rows[0]
        self.assertAlmostEqual(sum(parts[key] for key in
                                   ("student", "rater", "work", "residual")),
                               1.0, places=6)
        self.assertGreaterEqual(parts["student"], 0.5)
        self.assertGreaterEqual(parts["rater"], 0.4)

    def test_rater_scale_rows_are_labelled_with_component(self):
        dataset = scale_archive()
        rows = rater_scales(build_networks(dataset))
        self.assertEqual({row["rater_id"] for row in rows}, {"T1", "T2"})
        self.assertTrue(all(row["component_id"] for row in rows))


class QualificationAndRevisionTests(unittest.TestCase):
    def revision_archive(self):
        return make(
            skills=[skill("S")],
            students=[student("a"), student("b")],
            works=[work("W1", "a"), work("W2", "a"),
                   work("W3", "b"), work("W4", "b")],
            ratings=[
                rating("R1", "W1", "a", "S", "T1", 70, "2026-09-05"),
                rating("R2", "W1", "a", "S", "T2", 66, "2026-09-05"),
                rating("R3", "W2", "a", "S", "T1", 68, "2026-09-06"),
                rating("R4", "W2", "a", "S", "T2", 64, "2026-09-06"),
                rating("R5", "W3", "b", "S", "T1", 50, "2026-09-07"),
                rating("R6", "W3", "b", "S", "T2", 46, "2026-09-07"),
                rating("R7", "W4", "b", "S", "T1", 48, "2026-09-08"),
                rating("R8", "W4", "b", "S", "T2", 44, "2026-09-08"),
                # 同作品、同技能、同评分者的重评，是另一条判断。
                rating("R9", "W1", "a", "S", "T1", 30, "2026-10-01", "R1"),
            ],
            vacancies=[vacancy("V", "S:0.0")],
            sessions=[session("SE", "V", independent_slots=1,
                              supervised_slots=0)],
            registrations=[registration("G1", "V", "SE", "a", status="confirmed",
                                        decision_id="D1")],
            decisions=[decision("D1", "a", "V", ["R1", "R3"])])

    def test_revision_keeps_old_decision_but_new_evidence_excludes_it(self):
        dataset = self.revision_archive()
        networks = build_networks(dataset)
        trace = trace_decision(dataset, "D1")
        self.assertTrue(trace["contains_superseded_evidence"])
        first = next(row for row in trace["evidence"] if row["rating_id"] == "R1")
        self.assertFalse(first["still_current"])
        self.assertEqual(first["superseded_by"], "R9")
        self.assertEqual(first["chain"], ["R1", "R9"])

        current = qualification_evidence(dataset, networks, "a", "V")
        self.assertNotIn("R1", current["rating_ids"])
        self.assertIn("R9", current["rating_ids"])
        self.assertTrue(all(row["current"] for row in current["provenance"]))

    def test_confirmed_registration_is_retained_and_explained(self):
        dataset = self.revision_archive()
        networks = build_networks(dataset)
        plan = summarize_plan(dataset, networks)
        self.assertEqual(plan["confirmed_registration_ids"], ["G1"])
        retained = plan["retained"][0]
        self.assertTrue(retained["kept"])
        self.assertEqual(retained["outcome"], "confirmed")
        self.assertEqual(retained["revised_evidence"],
                         [{"rating_id": "R1", "chain": ["R1", "R9"]}])
        self.assertNotIn("R1", retained["current_rating_ids"])
        # 已确定报名没有被当成待审核项重新分配。
        self.assertNotIn("G1", [row["registration_id"]
                                for row in plan["requested"]])

    def test_superseded_evidence_alone_does_not_revoke_confirmation(self):
        dataset = self.revision_archive()
        networks = build_networks(dataset)
        standing = confirmed_standing(dataset, networks,
                                      dataset.registrations["G1"])
        self.assertTrue(standing["kept"])
        self.assertIn("不因后续评分被自动撤销", standing["note"])
        self.assertEqual(standing["revised_evidence"],
                         [{"rating_id": "R1", "chain": ["R1", "R9"]}])
        self.assertTrue(all(row["current"] for row in standing["current_provenance"]))


def two_strong_students(sessions, registrations, sessions_mentors=None):
    """两名能力明确高于要求的学员，用于检验带教容量核算。"""
    students, works, ratings = [], [], []
    for student_id in ("a", "b"):
        students.append(student(student_id))
        for index in (1, 2):
            work_id = f"W{student_id}{index}"
            works.append(work(work_id, student_id))
            for rater_id, severity in (("T1", 0.5), ("T2", -0.5)):
                score = round(from_logit(0.9 - severity, 0.0, 100.0), 2)
                ratings.append(rating(f"R{student_id}{index}{rater_id}", work_id,
                                      student_id, "S", rater_id, score))
    return make(
        skills=[skill("S")], students=students, works=works, ratings=ratings,
        vacancies=[vacancy("V", "S:-1.0")], sessions=sessions,
        mentors=sessions_mentors or [mentor("M1", capacity=1)],
        registrations=registrations)


class PlacementTests(unittest.TestCase):
    def qualified_archive(self):
        strong = {"q", "q2", "q3"}
        students, works, ratings = [], [], []
        for student_id in ("q", "q2", "q3", "x"):
            students.append(student(student_id))
            ability = 0.9 if student_id in strong else -0.9
            for index in (1, 2):
                work_id = f"W{student_id}{index}"
                works.append(work(work_id, student_id))
                for rater_id, severity in (("T1", 0.5), ("T2", -0.5)):
                    score = round(from_logit(ability - severity, 0.0, 100.0), 2)
                    ratings.append(rating(
                        f"R{student_id}{index}{rater_id}", work_id, student_id,
                        "S", rater_id, score))
        students.append(student("u"))
        works.append(work("Wu1", "u"))
        ratings.append(rating("Ru1", "Wu1", "u", "S2", "T9", 90))
        return make(
            skills=[skill("S"), skill("S2")],
            students=students, works=works, ratings=ratings,
            vacancies=[vacancy("V", "S:0.0"), vacancy("V2", "S2:0.0")],
            sessions=[session("SE-A", "V", start="09:00", end="11:00",
                              independent_slots=1, supervised_slots=1,
                              mentor_id="M1"),
                      session("SE-B", "V", start="10:00", end="12:00",
                              independent_slots=0, supervised_slots=1,
                              mentor_id="M1"),
                      session("SE-U", "V2", start="09:00", end="11:00",
                              independent_slots=1, supervised_slots=0)],
            mentors=[mentor("M1", capacity=1)],
            registrations=[
                registration("G1", "V", "SE-A", "q"),
                registration("G2", "V", "SE-A", "q2"),
                registration("G3", "V", "SE-B", "q3", mode="supervised"),
                registration("G4", "V", "SE-A", "x"),
                registration("G5", "V2", "SE-U", "u"),
                registration("G6", "V", "SE-B", "q2"),
            ])

    def plans(self):
        dataset = self.qualified_archive()
        networks = build_networks(dataset)
        plan = summarize_plan(dataset, networks)
        return {row["registration_id"]: row for row in plan["requested"]}, plan

    def test_qualified_request_gets_independent_slot(self):
        plans, _ = self.plans()
        self.assertEqual(plans["G1"]["recommendation"], "independent")

    def test_full_independent_slot_falls_back_to_supervised(self):
        plans, _ = self.plans()
        self.assertEqual(plans["G2"]["recommendation"], "supervised")
        self.assertEqual(plans["G2"]["mentor_id"], "M1")
        self.assertIn("独立席位已满", " ".join(plans["G2"]["reasons"]))

    def test_mentor_capacity_is_counted_across_overlapping_sessions(self):
        plans, plan = self.plans()
        self.assertEqual(plans["G3"]["recommendation"], "pending")
        self.assertIn("超出容量", " ".join(plans["G3"]["reasons"]))
        mentor_row = plan["mentors"][0]
        self.assertEqual(mentor_row["capacity"], 1)
        self.assertEqual(mentor_row["peak_concurrent"], 1)
        self.assertFalse(mentor_row["over_capacity"])

    def test_uncertain_or_unsupported_evidence_never_gets_independent_shift(self):
        plans, _ = self.plans()
        # G4 证据充分但低于要求；G5 完全没有可用评分。
        self.assertEqual(plans["G4"]["recommendation"], "pending")
        self.assertIn("必要技能证据未达到要求",
                      " ".join(plans["G4"]["reasons"]))
        self.assertEqual(plans["G5"]["recommendation"], "pending")
        self.assertIn("证据不足", " ".join(plans["G5"]["reasons"]))
        self.assertIn("暂缓判断", " ".join(plans["G5"]["reasons"]))

    def test_qualified_student_is_not_double_booked(self):
        plans, _ = self.plans()
        self.assertEqual(plans["G6"]["recommendation"], "pending")
        self.assertIn("学员在重叠时段已有已确定服务", " ".join(plans["G6"]["reasons"]))

    def test_plan_exposes_unfilled_slots_and_reasons(self):
        _, plan = self.plans()
        self.assertEqual(plan["unassigned_reasons"]["必要技能证据未达到要求"], 1)
        by_vacancy = {row["vacancy_id"]: row for row in plan["vacancies"]}
        self.assertEqual(by_vacancy["V"]["confirmed"], 0)
        self.assertEqual(by_vacancy["V"]["capacity"], 3)
        self.assertEqual(by_vacancy["V"]["independent"], 1)
        self.assertEqual(by_vacancy["V"]["supervised"], 1)
        self.assertEqual(by_vacancy["V"]["pending"], 3)
        self.assertEqual(by_vacancy["V"]["unfilled_slots"], 1)
        self.assertEqual(by_vacancy["V2"]["pending"], 1)
        self.assertIn("超出容量", " ".join(plan["unassigned_reasons"]))

    def test_mentor_capacity_counts_simultaneous_not_sequential_sessions(self):
        dataset = two_strong_students(
            sessions=[session("SE1", "V", start="09:00", end="11:00",
                              independent_slots=0, supervised_slots=1,
                              mentor_id="M1"),
                      session("SE2", "V", start="13:00", end="15:00",
                              independent_slots=0, supervised_slots=1,
                              mentor_id="M1")],
            registrations=[registration("G1", "V", "SE1", "a",
                                        mode="supervised"),
                           registration("G2", "V", "SE2", "b",
                                        mode="supervised")])
        plan = summarize_plan(dataset, build_networks(dataset))
        recommendations = {row["registration_id"]: row["recommendation"]
                           for row in plan["requested"]}
        self.assertEqual(recommendations["G1"], "supervised")
        # 不重叠的两个时段可以由同一位带教者先后承担。
        self.assertEqual(recommendations["G2"], "supervised")
        self.assertEqual(plan["mentors"][0]["peak_concurrent"], 1)

    def test_mentor_capacity_rejects_simultaneous_sessions(self):
        dataset = two_strong_students(
            sessions=[session("SE1", "V", start="09:00", end="11:00",
                              independent_slots=0, supervised_slots=1,
                              mentor_id="M1"),
                      session("SE2", "V", start="10:00", end="12:00",
                              independent_slots=0, supervised_slots=1,
                              mentor_id="M1")],
            registrations=[registration("G1", "V", "SE1", "a",
                                        mode="supervised"),
                           registration("G2", "V", "SE2", "b",
                                        mode="supervised")])
        plan = summarize_plan(dataset, build_networks(dataset))
        recommendations = {row["registration_id"]: row["recommendation"]
                           for row in plan["requested"]}
        self.assertEqual(recommendations["G1"], "supervised")
        self.assertEqual(recommendations["G2"], "pending")
        self.assertEqual(plan["mentors"][0]["peak_concurrent"], 1)
        self.assertEqual(plan["mentors"][0]["spare_concurrent"], 0)

    def test_mentor_with_capacity_two_takes_both_simultaneous_requests(self):
        dataset = two_strong_students(
            sessions=[session("SE1", "V", start="09:00", end="11:00",
                              independent_slots=0, supervised_slots=2,
                              mentor_id="M1")],
            registrations=[registration("G1", "V", "SE1", "a",
                                        mode="supervised"),
                           registration("G2", "V", "SE1", "b",
                                        mode="supervised")],
            sessions_mentors=[mentor("M1", capacity=2)])
        plan = summarize_plan(dataset, build_networks(dataset))
        self.assertEqual([row["recommendation"] for row in plan["requested"]],
                         ["supervised", "supervised"])
        self.assertEqual(plan["mentors"][0]["peak_concurrent"], 2)
        self.assertFalse(plan["mentors"][0]["over_capacity"])

    def test_confirmed_shift_consumes_capacity_without_being_rewritten(self):
        dataset = self.qualified_archive()
        # 已确定报名的学员 q3 先占用 SE-B 的带教名额。
        dataset.registrations["G7"] = dict(
            registration("G7", "V", "SE-B", "q3", mode="supervised",
                         mentor_id="M1", status="confirmed", decision_id="D7"))
        dataset.decisions["D7"] = decision(
            "D7", "q3", "V", ["Rq31T1"], decided_at="2026-10-01")
        dataset.registrations["G3"]["status"] = "withdrawn"
        dataset.registrations["G3"]["decision_id"] = ""
        networks = build_networks(dataset)
        plan = summarize_plan(dataset, networks)
        retained = {row["registration_id"] for row in plan["retained"]}
        self.assertIn("G7", retained)
        mentor_row = plan["mentors"][0]
        self.assertEqual(mentor_row["assigned"], 1)
        self.assertEqual(mentor_row["peak_concurrent"], 1)


class StorageAndReportTests(unittest.TestCase):
    def test_database_roundtrip_keeps_revision_chain(self):
        dataset = QualificationAndRevisionTests().revision_archive()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.db"
            counts = write_dataset(dataset, path)
            self.assertEqual(counts["ratings"], 9)
            rows = current_ratings(path)
            ids = {row[0] for row in rows}
            self.assertNotIn("R1", ids)
            self.assertIn("R9", ids)
            self.assertEqual(ids, dataset.current_rating_ids)

    def test_report_renders_all_sections(self):
        dataset = PlacementTests().qualified_archive()
        text = render(dataset, build_networks(dataset))
        for heading in ("可比评分网络覆盖", "评分者尺度", "原始平均分",
                        "岗位覆盖", "带教负担", "重评追溯"):
            self.assertIn(heading, text)
        self.assertIn("暂缓判断", text)


if __name__ == "__main__":
    unittest.main()
