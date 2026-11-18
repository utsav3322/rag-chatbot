"""Generate synthetic Q4 FY2025 earnings-call PDFs for the demo corpus.

The transcripts are fictional but written in the style of a real earnings call
(prepared remarks + analyst Q&A) with distinctive figures per company, so that
retrieval, citation, and per-user isolation are easy to demonstrate. Using
generated data keeps the repository self-contained and free of any third-party
copyright concerns.

Run:  python scripts/generate_sample_pdfs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# Make the package importable whether run from repo root or scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from docqa.config import PDF_DIR  # noqa: E402


# Each entry drives one earnings-call transcript. Figures are intentionally
# distinct across companies so cross-company leakage is obvious if it happens.
COMPANIES: list[dict[str, object]] = [
    {
        "key": "novatech",
        "name": "NovaTech Industries",
        "ticker": "NOVA",
        "sector": "enterprise cloud software",
        "ceo": "Maria Chen",
        "cfo": "David Okafor",
        "revenue": "$2.34 billion",
        "revenue_growth": "up 28% year over year",
        "net_income": "$412 million",
        "eps": "$1.87 per diluted share",
        "segments": (
            "Cloud Platform revenue was $1.62 billion, up 34%, and now represents "
            "69% of total revenue. Professional Services contributed $720 million, "
            "up 15%."
        ),
        "margin": "Non-GAAP operating margin expanded to 31%, up from 26% a year ago.",
        "guidance": (
            "For fiscal 2026 we expect full-year revenue of $10.4 to $10.6 billion, "
            "representing growth of roughly 24%, and non-GAAP operating margin of "
            "about 32%."
        ),
        "metric": (
            "Net revenue retention was 121%, and we closed 42 deals larger than "
            "$1 million in annual contract value."
        ),
        "qa": [
            (
                "On the strength of Cloud Platform growth and whether it is durable",
                "Cloud Platform growth of 34% was broad based. Our new AI copilot "
                "add-on reached $140 million in annualised recurring revenue within "
                "two quarters, and we see a long runway as customers consolidate "
                "onto the platform.",
            ),
            (
                "On operating margin and reinvestment",
                "We expanded operating margin to 31% while still investing in R&D, "
                "which grew 22%. We believe we can balance growth and profitability "
                "and exit fiscal 2026 around 32% to 33% margin.",
            ),
        ],
    },
    {
        "key": "helios",
        "name": "Helios Energy Corp",
        "ticker": "HELI",
        "sector": "renewable power generation",
        "ceo": "Priya Nair",
        "cfo": "Thomas Reed",
        "revenue": "$3.78 billion",
        "revenue_growth": "up 11% year over year",
        "net_income": "$305 million",
        "eps": "$2.14 per diluted share",
        "segments": (
            "Solar generation revenue was $2.10 billion, up 18%. Wind generation "
            "revenue was $1.21 billion, up 4%. Storage and grid services added "
            "$470 million, up 36%."
        ),
        "margin": "Adjusted EBITDA margin was 41%, broadly flat year over year.",
        "guidance": (
            "For fiscal 2026 we expect revenue of $4.0 to $4.2 billion and plan "
            "capital expenditure of $1.3 billion, weighted toward battery storage "
            "and transmission interconnection."
        ),
        "metric": (
            "We brought 1.4 gigawatts of new capacity online this quarter and our "
            "contracted backlog stands at 9.2 gigawatts."
        ),
        "qa": [
            (
                "On the impact of interconnection delays",
                "Interconnection queues remain the key bottleneck. We mitigated the "
                "impact by prioritising projects with signed grid agreements, and "
                "we still expect to energise 5 gigawatts across fiscal 2026.",
            ),
            (
                "On power purchase agreement pricing",
                "Average realised PPA pricing rose 6% as we re-contracted legacy "
                "solar at higher rates. Storage attach rates on new PPAs reached "
                "55%, which lifts both revenue and margin.",
            ),
        ],
    },
    {
        "key": "verdant",
        "name": "Verdant Foods",
        "ticker": "VERD",
        "sector": "packaged consumer foods",
        "ceo": "Sofia Alvarez",
        "cfo": "Kenji Watanabe",
        "revenue": "$1.96 billion",
        "revenue_growth": "up 6% year over year",
        "net_income": "$188 million",
        "eps": "$0.94 per diluted share",
        "segments": (
            "Plant-based products revenue was $640 million, up 19%. Snacks revenue "
            "was $820 million, up 3%. Beverages revenue was $500 million, down 2%."
        ),
        "margin": "Gross margin improved 150 basis points to 38%, helped by lower "
        "input costs for grains and packaging.",
        "guidance": (
            "For fiscal 2026 we expect organic revenue growth of 4% to 6% and "
            "adjusted earnings per share of $3.90 to $4.05 for the full year."
        ),
        "metric": (
            "Pricing contributed 3 points and volume contributed 3 points to "
            "growth, the first positive volume quarter in two years."
        ),
        "qa": [
            (
                "On volume recovery and promotional spend",
                "Volume turned positive this quarter as we moderated price "
                "increases and stepped up promotion in snacks. We expect volume to "
                "remain positive through fiscal 2026 without eroding gross margin.",
            ),
            (
                "On the beverages decline",
                "Beverages declined 2% as we exited low-margin private-label "
                "contracts. The remaining branded portfolio grew 4%, and we expect "
                "the segment to return to growth in the second half.",
            ),
        ],
    },
    {
        "key": "orion",
        "name": "Orion Logistics",
        "ticker": "ORIO",
        "sector": "freight and supply-chain logistics",
        "ceo": "Marcus Bell",
        "cfo": "Lena Fischer",
        "revenue": "$5.42 billion",
        "revenue_growth": "down 3% year over year",
        "net_income": "$221 million",
        "eps": "$1.05 per diluted share",
        "segments": (
            "Freight Forwarding revenue was $3.10 billion, down 8% on lower ocean "
            "rates. Contract Logistics revenue was $1.72 billion, up 9%. "
            "Last-Mile Delivery revenue was $600 million, up 14%."
        ),
        "margin": "Operating margin was 6.8%, down from 8.1% as freight rates "
        "normalised from pandemic-era highs.",
        "guidance": (
            "For fiscal 2026 we expect revenue of $5.5 to $5.8 billion and operating "
            "margin of 7% to 8% as volumes recover and the cost programme delivers "
            "$180 million of savings."
        ),
        "metric": (
            "We automated 14 distribution centres this year, improving throughput "
            "per labour hour by 17%."
        ),
        "qa": [
            (
                "On the freight-rate environment",
                "Ocean and air rates have largely normalised. We expect modest rate "
                "recovery in the back half of fiscal 2026 as capacity tightens, but "
                "our model is increasingly weighted to fee-based contract logistics.",
            ),
            (
                "On the cost-savings programme",
                "We are on track for $180 million of annualised savings from "
                "automation and network consolidation, with roughly half realised "
                "in the first half of fiscal 2026.",
            ),
        ],
    },
    {
        "key": "lumina",
        "name": "Lumina Pharmaceuticals",
        "ticker": "LUMA",
        "sector": "biopharmaceuticals",
        "ceo": "Dr. Ananya Rao",
        "cfo": "Gregory Santos",
        "revenue": "$4.11 billion",
        "revenue_growth": "up 17% year over year",
        "net_income": "$1.02 billion",
        "eps": "$3.42 per diluted share",
        "segments": (
            "Immunology franchise revenue was $2.05 billion, up 24%, led by our lead "
            "therapy Velora. Oncology revenue was $1.36 billion, up 12%. Rare "
            "Disease revenue was $700 million, up 9%."
        ),
        "margin": "Non-GAAP gross margin was 82%, and R&D expense was $980 million, "
        "or 24% of revenue.",
        "guidance": (
            "For fiscal 2026 we expect revenue of $4.6 to $4.8 billion, assuming the "
            "approval of Velora in a second indication in the second half of the "
            "year."
        ),
        "metric": (
            "Velora delivered $1.30 billion in quarterly revenue, and three Phase 3 "
            "readouts are expected over the next four quarters."
        ),
        "qa": [
            (
                "On the Velora label expansion timeline",
                "We filed the supplemental application in the first quarter and "
                "expect a regulatory decision in the second half of fiscal 2026. A "
                "positive outcome would roughly double the addressable population.",
            ),
            (
                "On pipeline and R&D spending",
                "R&D was 24% of revenue, funding three Phase 3 programmes. We expect "
                "to sustain R&D intensity near 23% to 25% as the oncology pipeline "
                "advances toward registration.",
            ),
        ],
    },
]


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontSize=18, spaceAfter=6
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Heading2"], fontSize=12, textColor="#555555"
        ),
        "heading": ParagraphStyle(
            "H", parent=base["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6
        ),
        "speaker": ParagraphStyle(
            "Speaker", parent=base["Heading3"], fontSize=11, spaceBefore=10
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=10.5,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
    }


def _build_flowables(c: dict[str, object], styles: dict[str, ParagraphStyle]) -> list:
    S = styles
    f: list = []
    f.append(Paragraph(f"{c['name']} ({c['ticker']})", S["title"]))
    f.append(
        Paragraph("Q4 Fiscal Year 2025 Earnings Conference Call — Transcript", S["subtitle"])
    )
    f.append(Spacer(1, 10))

    f.append(Paragraph("Operator", S["speaker"]))
    f.append(
        Paragraph(
            f"Good afternoon and welcome to the {c['name']} fourth-quarter fiscal "
            f"2025 earnings conference call. Joining us today are Chief Executive "
            f"Officer {c['ceo']} and Chief Financial Officer {c['cfo']}. "
            "All participants are in a listen-only mode. After prepared remarks "
            "there will be a question-and-answer session.",
            S["body"],
        )
    )

    f.append(Paragraph(f"{c['ceo']} — Chief Executive Officer", S["speaker"]))
    f.append(
        Paragraph(
            f"Thank you, operator, and good afternoon everyone. {c['name']} is a "
            f"leader in {c['sector']}, and I'm pleased to report a strong close to "
            f"fiscal 2025. Total revenue for the fourth quarter was {c['revenue']}, "
            f"{c['revenue_growth']}. {c['metric']} These results reflect consistent "
            "execution and healthy demand across our portfolio.",
            S["body"],
        )
    )
    f.append(
        Paragraph(
            f"By segment, {c['segments']} We continue to invest behind our strongest "
            "growth drivers while maintaining discipline on cost.",
            S["body"],
        )
    )

    f.append(Paragraph(f"{c['cfo']} — Chief Financial Officer", S["speaker"]))
    f.append(
        Paragraph(
            f"Thank you, {str(c['ceo']).split()[0]}. Let me turn to the financial "
            f"details. Fourth-quarter revenue was {c['revenue']}, {c['revenue_growth']}. "
            f"Net income for the quarter was {c['net_income']}, and diluted earnings "
            f"per share were {c['eps']}. {c['margin']}",
            S["body"],
        )
    )
    f.append(
        Paragraph(
            f"Turning to guidance. {c['guidance']} We remain focused on durable, "
            "profitable growth and on returning capital to shareholders over time.",
            S["body"],
        )
    )

    f.append(Paragraph("Questions and Answers", S["heading"]))
    for question, response in c["qa"]:  # type: ignore[attr-defined]
        f.append(Paragraph("Analyst", S["speaker"]))
        f.append(Paragraph(f"A question {question}.", S["body"]))
        f.append(Paragraph(f"{c['cfo']} — Chief Financial Officer", S["speaker"]))
        f.append(Paragraph(response, S["body"]))

    f.append(Paragraph("Operator", S["speaker"]))
    f.append(
        Paragraph(
            "This concludes today's conference call. Thank you for your participation; "
            "you may now disconnect.",
            S["body"],
        )
    )
    return f


def generate(output_dir: Path = PDF_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    written: list[Path] = []
    for company in COMPANIES:
        path = output_dir / f"{company['key']}.pdf"
        doc = SimpleDocTemplate(
            str(path),
            pagesize=LETTER,
            topMargin=0.9 * inch,
            bottomMargin=0.9 * inch,
            leftMargin=1.0 * inch,
            rightMargin=1.0 * inch,
            title=f"{company['name']} Q4 FY2025 Earnings Call",
        )
        doc.build(_build_flowables(company, styles))
        written.append(path)
        print(f"  wrote {path.relative_to(output_dir.parent.parent)}")
    return written


if __name__ == "__main__":
    print(f"Generating {len(COMPANIES)} earnings-call PDFs into {PDF_DIR} ...")
    generate()
    print("Done.")
