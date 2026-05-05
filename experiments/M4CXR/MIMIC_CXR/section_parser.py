"""Section parser for MIMIC-CXR reports.

Adapted from the public MIMIC-CXR / M4CXR preprocessing utilities:
- https://github.com/MIT-LCP/mimic-cxr
- https://github.com/deepnoid-ai/M4CXR-TNNLS
"""

from __future__ import annotations

import re


def section_text(text: str):
    """Split a report into coarse sections based on all-caps headers."""
    p_section = re.compile(r"\n ([A-Z ()/,-]+):\s", re.DOTALL)

    sections = []
    section_names = []
    section_idx = []

    idx = 0
    match = p_section.search(text, idx)

    if match:
        sections.append(text[0 : match.start(1)])
        section_names.append("preamble")
        section_idx.append(0)

        while match:
            current_section = match.group(1).lower()
            idx_start = match.end()
            idx_skip = text[idx_start:].find("\n")
            if idx_skip == -1:
                idx_skip = 0

            match = p_section.search(text, idx_start + idx_skip)
            idx_end = len(text) if match is None else match.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)
    else:
        sections.append(text)
        section_names.append("full report")
        section_idx.append(0)

    section_names = normalize_section_names(section_names)

    for i in reversed(range(len(section_names))):
        if section_names[i] in ("impression", "findings") and sections[i].strip() == "":
            sections.pop(i)
            section_names.pop(i)
            section_idx.pop(i)

    if ("impression" not in section_names) and ("findings" not in section_names):
        if "\n \n" in sections[-1]:
            sections.append("\n \n".join(sections[-1].split("\n \n")[1:]))
            sections[-2] = sections[-2].split("\n \n")[0]
            section_names.append("last_paragraph")
            section_idx.append(section_idx[-1] + len(sections[-2]))

    return sections, section_names, section_idx


def normalize_section_names(section_names: list[str]) -> list[str]:
    section_names = [s.lower().strip() for s in section_names]

    frequent_sections = {
        "preamble": "preamble",
        "impression": "impression",
        "comparison": "comparison",
        "indication": "indication",
        "findings": "findings",
        "examination": "examination",
        "technique": "technique",
        "history": "history",
        "comparisons": "comparison",
        "clinical history": "history",
        "reason for examination": "indication",
        "notification": "notification",
        "reason for exam": "indication",
        "clinical information": "history",
        "exam": "examination",
        "clinical indication": "indication",
        "conclusion": "impression",
        "chest, two views": "findings",
        "recommendation(s)": "recommendations",
        "type of examination": "examination",
        "reference exam": "comparison",
        "patient history": "history",
        "addendum": "addendum",
        "comparison exam": "comparison",
        "date": "date",
        "comment": "comment",
        "findings and impression": "impression",
        "wet read": "wet read",
        "comparison film": "comparison",
        "recommendations": "recommendations",
        "findings/impression": "impression",
        "pfi": "history",
        "recommendation": "recommendations",
        "wetread": "wet read",
        "ndication": "impression",
        "impresson": "impression",
        "imprression": "impression",
        "imoression": "impression",
        "impressoin": "impression",
        "imprssion": "impression",
        "impresion": "impression",
        "imperssion": "impression",
        "mpression": "impression",
        "impession": "impression",
        "findings/ impression": "impression",
        "finding": "findings",
        "findins": "findings",
        "findindgs": "findings",
        "findgings": "findings",
        "findngs": "findings",
        "findnings": "findings",
        "finidngs": "findings",
        "idication": "indication",
        "reference findings": "findings",
        "comparision": "comparison",
        "comparsion": "comparison",
        "comparrison": "comparison",
        "comparisions": "comparison",
    }

    p_findings = [
        "chest",
        "portable",
        "pa and lateral",
        "lateral and pa",
        "ap and lateral",
        "lateral and ap",
        "frontal and",
        "two views",
        "frontal view",
        "pa view",
        "ap view",
        "one view",
        "lateral view",
        "bone window",
        "frontal upright",
        "frontal semi-upright",
        "ribs",
        "pa and lat",
    ]
    p_findings_re = re.compile("({})".format("|".join(p_findings)))

    main_sections = ["impression", "findings", "history", "comparison", "addendum"]
    for i, name in enumerate(section_names):
        if name in frequent_sections:
            section_names[i] = frequent_sections[name]
            continue

        for main_name in main_sections:
            if main_name in name:
                section_names[i] = main_name
                break
        else:
            match = p_findings_re.search(name)
            if match is not None:
                section_names[i] = "findings"

    return section_names


def custom_mimic_cxr_rules():
    custom_section_names = {
        "s50913680": "recommendations",
        "s59363654": "examination",
        "s59279892": "technique",
        "s59768032": "recommendations",
        "s57936451": "indication",
        "s50058765": "indication",
        "s53356173": "examination",
        "s53202765": "technique",
        "s50808053": "technique",
        "s51966317": "indication",
        "s50743547": "examination",
        "s56451190": "note",
        "s59067458": "recommendations",
        "s59215320": "examination",
        "s55124749": "indication",
        "s54365831": "indication",
        "s59087630": "recommendations",
        "s58157373": "recommendations",
        "s56482935": "recommendations",
        "s58375018": "recommendations",
        "s54654948": "indication",
        "s55157853": "examination",
        "s51491012": "history",
    }

    custom_indices = {
        "s50525523": [201, 349],
        "s57564132": [233, 554],
        "s59982525": [313, 717],
        "s53488209": [149, 475],
        "s54875119": [234, 988],
        "s50196495": [59, 399],
        "s56579911": [59, 218],
        "s52648681": [292, 631],
        "s59889364": [172, 453],
        "s53514462": [73, 377],
        "s59505494": [59, 450],
        "s53182247": [59, 412],
        "s51410602": [47, 320],
        "s56412866": [522, 822],
        "s54986978": [59, 306],
        "s59003148": [262, 505],
        "s57150433": [61, 394],
        "s56760320": [219, 457],
        "s59562049": [158, 348],
        "s52674888": [145, 296],
        "s55258338": [192, 568],
        "s59330497": [140, 655],
        "s52119491": [179, 454],
        "s58235663": [0, 0],
        "s50798377": [0, 0],
        "s54168089": [0, 0],
        "s53071062": [0, 0],
        "s56724958": [0, 0],
        "s54231141": [0, 0],
        "s53607029": [0, 0],
    }
    return custom_section_names, custom_indices
