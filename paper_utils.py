
import fitz
import re


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):
    """
    Extract text from a PDF file-like object.
    """

    try:
        file_bytes = uploaded_file.read()

        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        pages = []

        for page_number, page in enumerate(document):

            page_text = page.get_text("text")

            pages.append(
                f"\n--- PAGE {page_number + 1} ---\n"
                f"{page_text}"
            )

        full_text = "\n".join(pages)

        page_count = len(document)

        document.close()

        return {
            "status": "success",
            "text": full_text,
            "page_count": page_count
        }

    except Exception as exc:

        return {
            "status": "error",
            "error": str(exc)
        }


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """
    Clean extraction artifacts without changing
    scientific meaning.
    """

    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# VALIDATION
# =========================================================

def validate_paper_text(text):
    """
    Check whether enough readable PDF text exists.
    """

    if not text or not text.strip():

        return (
            False,
            "No readable text was found in this PDF."
        )

    if len(text.strip()) < 1000:

        return (
            False,
            "The document does not contain enough readable "
            "research content."
        )

    return True, None


# =========================================================
# SECTION DEFINITIONS
# =========================================================

SECTION_PATTERNS = {

    "abstract": [
        "abstract"
    ],

    "introduction": [
        "introduction",
        "background"
    ],

    "literature_review": [
        "literature review",
        "related work",
        "related works"
    ],

    "methodology": [
        "methodology",
        "methods",
        "materials and methods",
        "research methodology",
        "experimental setup",
        "experimental method"
    ],

    "results": [
        "results",
        "findings",
        "experimental results"
    ],

    "discussion": [
        "discussion"
    ],

    "limitations": [
        "limitations",
        "study limitations"
    ],

    "conclusion": [
        "conclusion",
        "conclusions"
    ],

    "future_work": [
        "future work",
        "future research",
        "future directions"
    ],

    "references": [
        "references",
        "bibliography"
    ],
}


# =========================================================
# SECTION DETECTION
# =========================================================

def detect_sections(text):
    """
    Detect common scientific section headings.
    """

    detected = {}

    lines = text.splitlines()

    for index, line in enumerate(lines):

        cleaned_line = (
            line
            .strip()
            .lower()
        )

        if not cleaned_line:
            continue

        # Remove numbering such as:
        # 1 Introduction
        # 2.1 Methodology
        # III. Results is not aggressively handled
        cleaned_line = re.sub(
            r"^\d+(\.\d+)*\.?\s*",
            "",
            cleaned_line
        )

        cleaned_line = cleaned_line.rstrip(":")

        for section, patterns in SECTION_PATTERNS.items():

            if any(
                cleaned_line == pattern
                for pattern in patterns
            ):

                if section not in detected:
                    detected[section] = index

    return detected


# =========================================================
# EXTRACT SECTION
# =========================================================

def extract_section_text(
    text,
    detected_sections,
    section_name
):
    """
    Extract text belonging to one detected section.
    """

    if section_name not in detected_sections:
        return ""

    lines = text.splitlines()

    start_line = detected_sections[
        section_name
    ]

    ordered_sections = sorted(
        detected_sections.items(),
        key=lambda item: item[1]
    )

    end_line = len(lines)

    for _, line_number in ordered_sections:

        if line_number > start_line:

            end_line = line_number

            break

    return "\n".join(
        lines[start_line:end_line]
    ).strip()


# =========================================================
# FRONT MATTER / TITLE AREA
# =========================================================

def extract_front_matter(
    text,
    detected_sections,
    max_lines=40
):
    """
    Preserve the text appearing before the Abstract.

    This commonly includes:
    title
    authors
    affiliation
    journal metadata
    """

    lines = text.splitlines()

    if "abstract" in detected_sections:

        abstract_line = detected_sections[
            "abstract"
        ]

        beginning = max(
            0,
            abstract_line - max_lines
        )

        return "\n".join(
            lines[
                beginning:abstract_line
            ]
        ).strip()

    return "\n".join(
        lines[:max_lines]
    ).strip()


# =========================================================
# SIMPLE TITLE HEURISTIC
# =========================================================

def detect_probable_title(
    text,
    detected_sections
):
    """
    Estimate a probable paper title from the front matter.
    This is only used for UI preview.
    Gemini performs the final title interpretation.
    """

    front_matter = extract_front_matter(
        text,
        detected_sections,
        max_lines=25
    )

    lines = [
        line.strip()
        for line in front_matter.splitlines()
        if line.strip()
    ]

    ignored_words = {
        "abstract",
        "doi",
        "received",
        "accepted",
        "published",
        "copyright"
    }

    candidates = []

    for line in lines:

        lowered = line.lower()

        if any(
            word in lowered
            for word in ignored_words
        ):
            continue

        word_count = len(
            line.split()
        )

        # Research titles are often neither
        # one word nor extremely long.
        if 4 <= word_count <= 30:

            candidates.append(line)

    if candidates:

        # Prefer longer descriptive lines.
        candidates.sort(
            key=len,
            reverse=True
        )

        return candidates[0]

    return "Title will be identified during AI analysis"


# =========================================================
# PREPARE GEMINI CONTEXT
# =========================================================

def prepare_paper_context(
    text,
    detected_sections,
    max_chars=45000
):
    """
    Prepare the scientifically important paper content
    while excluding the References section.

    This keeps Gemini requests faster and more focused.
    """

    priority_sections = [
        "abstract",
        "introduction",
        "literature_review",
        "methodology",
        "results",
        "discussion",
        "limitations",
        "conclusion",
        "future_work"
    ]

    context_parts = []

    # Important because the title normally occurs
    # before the Abstract.
    front_matter = extract_front_matter(
        text,
        detected_sections
    )

    if front_matter:

        context_parts.append(
            "### FRONT MATTER / POSSIBLE TITLE\n"
            + front_matter
        )

    for section in priority_sections:

        section_text = extract_section_text(
            text,
            detected_sections,
            section
        )

        if section_text:

            context_parts.append(
                f"\n### {section.upper()}\n"
                f"{section_text}"
            )

    if context_parts:

        context = "\n".join(
            context_parts
        )

    else:

        # Fallback when headings are not detected.
        context = text

    if len(context) > max_chars:

        context = context[:max_chars]

        context += (
            "\n\n[NOTE: The supplied paper context "
            "was shortened because of input length limits.]"
        )

    return context
