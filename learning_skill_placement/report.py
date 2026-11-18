"""命令行报告：把档案转成可核对的运营摘要。

输出刻意分成“证据是否可比”“单个学员证据”“岗位与排班”三块，
避免把估计值直接包装成资格或排班承诺。
"""

import argparse
import tempfile
from pathlib import Path

from .analysis import raw_vs_adjusted, rater_scales
from .network import build_networks, coverage
from .placement import summarize_plan
from .qualification import trace_decision
from .records import load_dataset
from .storage import current_ratings, write_dataset


def _percent(value):
    return "   —" if value is None else f"{value * 100:5.1f}%"


def _interval(interval):
    if not interval:
        return "无区间"
    return f"[{interval[0]:+.2f}, {interval[1]:+.2f}]"


def render(dataset, networks):
    lines = []
    lines.append("=" * 78)
    lines.append("一、可比评分网络覆盖")
    lines.append("=" * 78)
    for skill in coverage(networks):
        skill_name = dataset.skills[skill["skill_id"]]["name"]
        lines.append(f"{skill['skill_id']} {skill_name}：覆盖 {skill['students']} 位学员，"
                     f"{len(skill['components'])} 个可比评分组")
        for component in skill["components"]:
            parts = component["decomposition"]
            lines.append(
                f"  - {component['component_id']}：{component['students']} 学员 / "
                f"{component['raters']} 评分者 / {component['works']} 作品 / "
                f"{component['judgments']} 条评分"
                f"{'' if component['converged'] else '（未收敛）'}")
            lines.append(
                f"    变异份额：学员 {_percent(parts['student'])}　"
                f"作品 {_percent(parts['work'])}　"
                f"评分者 {_percent(parts['rater'])}　"
                f"残余判断 {_percent(parts['residual'])}")
    lines.append("  不同可比评分组之间不排统一名次；组内排序只在本组内有效。")

    lines.append("")
    lines.append("=" * 78)
    lines.append("二、评分者尺度（越大越严格，已扣除组内平均）")
    lines.append("=" * 78)
    for row in rater_scales(networks):
        lines.append(f"  {row['component_id']} {row['rater_id']}："
                     f"{row['severity']:+.2f}（{row['judgments']} 条评分）")

    lines.append("")
    lines.append("=" * 78)
    lines.append("三、原始平均分与调整后能力（logit 尺度）")
    lines.append("=" * 78)
    lines.append(f"  {'学员':<8}{'技能':<9}{'条数':>4}{'作品':>4}"
                 f"{'原始均分':>9}{'调整后':>9}  区间{'':<12}证据")
    for row in raw_vs_adjusted(dataset, networks):
        evidence = "可用" if row["confidence"] == "usable" else "不确定"
        lines.append(
            f"  {row['student_id']:<8}{row['skill_id']:<9}{row['judgments']:>4}"
            f"{row['works']:>4}{row['raw_mean']:>9.1f}{row['adjusted']:>9.2f}  "
            f"{_interval(row['interval']):<20}{evidence}")
    lines.append("  评分组只有一位评分者或一位学员时，学员能力与尺度不可分离，"
                 "只能给出不确定。")

    plan = summarize_plan(dataset, networks)
    lines.append("")
    lines.append("=" * 78)
    lines.append("四、岗位覆盖与排班建议")
    lines.append("=" * 78)
    for vacancy in plan["vacancies"]:
        lines.append(
            f"  {vacancy['vacancy_id']} {dataset.vacancies[vacancy['vacancy_id']]['title']}："
            f"{vacancy['sessions']} 个时段，容量 {vacancy['capacity']}，"
            f"已确定 {vacancy['confirmed']}，"
            f"独立 {vacancy['independent']}，带教 {vacancy['supervised']}，"
            f"暂缓 {vacancy['pending']}，空余 {vacancy['unfilled_slots']}")
    lines.append("")
    for entry in plan["requested"]:
        label = {"independent": "独立上岗", "supervised": "带教上岗",
                 "pending": "暂缓判断"}[entry["recommendation"]]
        mentor = f"，带教者 {entry['mentor_id']}" if entry["mentor_id"] else ""
        lines.append(f"  {entry['registration_id']} {entry['student_id']} "
                     f"{entry['vacancy_id']} {entry['session_id']}：{label}{mentor}")
        for reason in entry["reasons"]:
            lines.append(f"      · {reason}")

    lines.append("")
    lines.append("=" * 78)
    lines.append("五、带教负担（按同时发生核算）")
    lines.append("=" * 78)
    for mentor in plan["mentors"]:
        lines.append(
            f"  {mentor['mentor_id']} {dataset.mentors[mentor['mentor_id']]['name']}："
            f"容量 {mentor['capacity']}，已安排 {mentor['assigned']}，"
            f"同时最多 {mentor['peak_concurrent']}，余 {mentor['spare_concurrent']}"
            f"{'（超出容量）' if mentor['over_capacity'] else ''}")
    if plan["unassigned_reasons"]:
        lines.append("  未安排原因（按条数，同一报名可有多条）：")
        for reason, count in plan["unassigned_reasons"].items():
            lines.append(f"    - {reason}（{count}）")

    lines.append("")
    lines.append("=" * 78)
    lines.append("六、已确定报名的依据与重评追溯")
    lines.append("=" * 78)
    if not plan["retained"]:
        lines.append("  没有已确定报名。")
    for row in plan["retained"]:
        lines.append(f"  {row['registration_id']}：当时结论 {row['outcome']}；{row['note']}")
        lines.append(f"    当时理由：{row['reason']}")
        for revised in row["revised_evidence"]:
            lines.append(f"    依据 {revised['rating_id']} 已被重评接替："
                         f"{' → '.join(revised['chain'])}")
        lines.append(f"    若按现行评分重算：{row['current_recommendation']}；"
                     f"现行证据 {', '.join(row['current_rating_ids']) or '无'}")
        for reason in row["current_reasons"]:
            lines.append(f"      · {reason}")
    for decision in sorted(dataset.decisions):
        trace = trace_decision(dataset, decision)
        if trace["contains_superseded_evidence"]:
            lines.append(f"  {decision} 含有已被替换的依据，旧资格保留但不再"
                         f"作为新推荐证据。")

    lines.append("")
    lines.append("=" * 78)
    lines.append("七、口径")
    lines.append("=" * 78)
    settings = dataset.settings
    lines.append(f"  置信度 {settings.confidence:.0%}；最低证据 "
                 f"{settings.min_judgments} 条评分且 {settings.min_works} 件作品；"
                 f"评分组至少 {settings.min_component_raters} 位评分者、"
                 f"{settings.min_component_students} 位学员。")
    lines.append("  能力估计来自现行评分；被替换的评分只用于解释旧资格，"
                 "不进入新推荐。")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None,
                        help="档案目录（默认仓库内 data/）")
    parser.add_argument("--database", default=None,
                        help="SQLite 输出路径（默认临时文件，不保留）")
    args = parser.parse_args(argv)

    root = Path(args.data) if args.data else Path(__file__).parent.parent / "data"
    dataset = load_dataset(root)
    networks = build_networks(dataset)

    if args.database:
        counts = write_dataset(dataset, Path(args.database))
        database = Path(args.database)
        print(f"已写入 {database}："
              + "，".join(f"{table} {count}" for table, count in counts.items()))
        print(f"现行评分 {len(current_ratings(database))} 条")
        print()
        print(render(dataset, networks))
        return

    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "archive.db"
        write_dataset(dataset, database)
        assert len(current_ratings(database)) == len(dataset.current_rating_ids)
    print(render(dataset, networks))


if __name__ == "__main__":
    main()
