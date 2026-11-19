"""Example job postings for CLI and frontend input forms."""
SCENARIOS = {
    "suspicious": {
        "label": "Payment request example",
        "posting": {
            "title": "Data Entry Assistant",
            "description": "Immediate start. To secure this position, you must pay an application fee of $300 before an interview. Apply now.",
            "company_profile": None,
            "salary_range": "50000-65000",
        },
    },
    "ordinary": {
        "label": "Ordinary wording / possible false alarm",
        "posting": {
            "title": "Data Entry Assistant",
            "description": "Maintain customer records and check data accuracy. Interviews take place at our office. There is no application fee.",
            "company_profile": "An administrative services company supporting local businesses.",
            "salary_range": "30000-38000",
        },
    },
    "missing_info": {
        "label": "Limited information example",
        "posting": {
            "title": "Office Assistant",
            "description": "Assist the team with routine office tasks. Contact the recruiter for details.",
            "company_profile": None,
            "salary_range": None,
        },
    },
}
