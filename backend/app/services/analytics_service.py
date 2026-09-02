"""
Analytics service: computes the aggregate numbers the Dashboard needs
(brief 4.3 minimum widgets: student count, outcome distribution, high-risk
count, risk-by-course breakdown, highest-risk students table).

All computed from the same precomputed roster used by /students — one
source of truth, no separate "dashboard data" that could drift out of sync
with the "students list" data.
"""
from collections import Counter, defaultdict

from app.services.students_service import get_students


def get_summary():
    students = get_students()
    total = len(students)

    outcome_counts = Counter(s["prediction"] for s in students)
    risk_counts = Counter(s["risk_band"] for s in students)
    high_risk_count = risk_counts.get("High", 0)

    # Risk band breakdown per course — brief 4.3: "Risk/outcome by course
    # or a meaningful categorical feature"
    risk_by_course = defaultdict(lambda: Counter())
    for s in students:
        risk_by_course[s["course"]][s["risk_band"]] += 1
    risk_by_course_list = [
        {"course": course, **dict(counts)}
        for course, counts in sorted(risk_by_course.items())
    ]

    # Highest-risk students table — brief 4.3: "table of highest-risk
    # students sorted by model probability"
    highest_risk = sorted(
        students, key=lambda s: s["probabilities"].get("Dropout", 0), reverse=True
    )[:10]
    highest_risk_summary = [
        {
            "student_id": s["student_id"],
            "course": s["course"],
            "prediction": s["prediction"],
            "risk_band": s["risk_band"],
            "dropout_probability": s["probabilities"].get("Dropout", 0),
        }
        for s in highest_risk
    ]

    return {
        "total_students": total,
        "outcome_distribution": dict(outcome_counts),
        "risk_distribution": dict(risk_counts),
        "high_risk_count": high_risk_count,
        "high_risk_percentage": round(100 * high_risk_count / total, 1) if total else 0,
        "risk_by_course": risk_by_course_list,
        "highest_risk_students": highest_risk_summary,
    }
