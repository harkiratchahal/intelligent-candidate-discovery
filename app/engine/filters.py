from app.db.models import Candidates, Jobs


def apply(candidates: list[Candidates], job: Jobs) -> list[Candidates]:
    eligible = []

    for candidate in candidates:
        # Rule 1 — experience check
        if candidate.years_experience < job.min_experience:
            continue

        # Rule 2 — certification check (only if job actually requires certs)
        if job.required_certs:
            candidate_certs = candidate.certifications or []
            # every required cert must exist in candidate's cert list
            if not all(cert in candidate_certs for cert in job.required_certs):
                continue

        eligible.append(candidate)

    return eligible