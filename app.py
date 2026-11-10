"""
app.py - Streamlit UI for the Stale-Answer Detector.

Lets users paste an LLM-generated answer (or pick a demo example from the
sidebar), extracts potentially time-sensitive claims, verifies them against
live Tavily web search results, displays per-claim freshness cards with an
overall score, and optionally rewrites the answer with current information.
"""

import re

import streamlit as st

from extractor import extract_claims
from verifier import generate_corrected_answer, verify_claims

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stale-Answer Detector",
    page_icon="🕵️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Demo examples for the sidebar
# ---------------------------------------------------------------------------
DEMO_EXAMPLES = {
    "1. AI Models & Python Stack (Versions + Pricing)": (
        "OpenAI's flagship API model is GPT-4 Turbo, which costs $10 per million "
        "input tokens and $30 per million output tokens. For backend development, "
        "Python 3.11 is the latest stable release, and LangChain just released "
        "v0.1 as its first stable version."
    ),
    "2. Tech Companies & Subscriptions (Company + Pricing)": (
        "Linda Yaccarino is the CEO of X (formerly Twitter), while Sam Altman "
        "leads OpenAI, which reached a valuation of $86 billion in early 2024. "
        "For streaming entertainment, Netflix's ad-free Standard plan in the "
        "United States costs $15.49 per month."
    ),
    "3. AI Regulations & Dev Tools (Regulation + Pricing)": (
        "The European Union's AI Act is currently still a draft proposal and has "
        "not yet been formally passed into law. Under GDPR, maximum fines can "
        "reach up to 4% of a company's annual global turnover. Meanwhile, GitHub "
        "Copilot Individual costs $10 per month with no free tier available."
    ),
}

# Map claim status values to Streamlit badge labels and colors.
STATUS_STYLE = {
    "fresh": ("FRESH", "green"),
    "uncertain": ("UNCERTAIN", "yellow"),
    "outdated": ("OUTDATED", "red"),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def escape_markdown(text: str) -> str:
    """
    Escape unescaped '$' characters so Streamlit does not treat prices like
    '$10 ... $30' as LaTeX math expressions.
    """
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def load_demo_example(example_text: str) -> None:
    """
    Sidebar button callback: fills the text area and clears prior results.
    """
    st.session_state["answer_input"] = example_text
    st.session_state["results"] = None
    st.session_state["checked_answer"] = ""
    st.session_state["extract_model"] = None
    st.session_state["corrected_answer"] = None
    st.session_state["corrected_model"] = None


def init_session_state() -> None:
    """Initialize required keys in Streamlit session_state."""
    defaults = {
        "answer_input": "",
        "results": None,
        "checked_answer": "",
        "extract_model": None,
        "corrected_answer": None,
        "corrected_model": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_overall_badge(overall_score: int) -> tuple[str, str]:
    """Return a human-readable label and badge color for the overall score."""
    if overall_score >= 75:
        return "MOSTLY FRESH", "green"
    if overall_score >= 45:
        return "NEEDS REVIEW", "yellow"
    return "STALE / OUTDATED", "red"


def render_claim_card(index: int, item: dict) -> None:
    """Render a single per-claim verification card."""
    status = item.get("status", "uncertain")
    label, color = STATUS_STYLE.get(status, ("UNCERTAIN", "yellow"))
    score = item.get("score", 50)
    category = item.get("category", "other")

    with st.container(border=True):
        # Top row: status badge + category on the left, score on the right.
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.badge(label, color=color)
            model_name = item.get("model")
            model_suffix = f"  •  Answered by: {model_name}" if model_name else ""
            st.caption(f"Claim #{index}  •  Category: `{category}`{model_suffix}")
        with col_right:
            st.markdown(f"**Score: {score} / 100**")

        st.markdown(f"**Claim:** {escape_markdown(item.get('claim', ''))}")

        # Surface any per-claim Tavily or JSON parsing warning clearly.
        if item.get("error"):
            st.warning(f"Verification issue: {item['error']}")

        st.markdown(
            f"**Current Info:** {escape_markdown(item.get('current_info', ''))}"
        )
        st.markdown(f"**Reason:** {escape_markdown(item.get('reason', ''))}")

        sources = item.get("sources", [])
        if sources:
            st.markdown("**Sources:**")
            for src in sources:
                title = escape_markdown(src.get("title") or src.get("url", "Source"))
                url = src.get("url", "")
                if url:
                    st.markdown(f"- [{title}]({url})")
        else:
            st.caption("No source links available.")


# ---------------------------------------------------------------------------
# Main application layout
# ---------------------------------------------------------------------------
init_session_state()

# Sidebar with 3 demo examples
with st.sidebar:
    st.header("Demo Examples")
    st.write("Click an example below to load it into the text box:")

    for title, text in DEMO_EXAMPLES.items():
        st.button(
            title,
            on_click=load_demo_example,
            args=(text,),
            use_container_width=True,
        )

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Extracts time-sensitive claims (prices, versions, laws, companies, stats).\n"
        "2. Searches 5 live web sources per claim via Tavily.\n"
        "3. Judges freshness against today's date and shows sources."
    )

st.title("Stale-Answer Detector")
st.write(
    "Paste an LLM-generated answer below to detect outdated facts, verify them "
    "against live web sources, and rewrite the answer with current information."
)

# Text area bound to session_state so sidebar demo buttons update it cleanly.
st.text_area(
    "LLM-generated answer",
    key="answer_input",
    height=180,
    placeholder=(
        "Paste an AI-generated answer here (or choose one of the 3 demo "
        "examples from the sidebar)..."
    ),
)

check_clicked = st.button("Check Freshness", type="primary")

# Run extraction and live verification when the user clicks "Check Freshness".
if check_clicked:
    answer_text = st.session_state["answer_input"].strip()
    if not answer_text:
        st.warning("Please paste an LLM-generated answer (or pick a sidebar demo) first.")
    else:
        st.session_state["results"] = None
        st.session_state["extract_model"] = None
        st.session_state["corrected_answer"] = None
        st.session_state["corrected_model"] = None

        try:
            with st.spinner("Extracting claims and verifying against live web sources..."):
                claims, extract_model = extract_claims(answer_text)
                st.session_state["extract_model"] = extract_model

                if not claims:
                    st.session_state["results"] = []
                    st.session_state["checked_answer"] = answer_text
                else:
                    progress_bar = st.progress(
                        0,
                        text=f"Found {len(claims)} claim(s). Searching live web sources...",
                    )

                    def update_progress(idx: int, total: int, _result: dict) -> None:
                        progress_bar.progress(
                            idx / total,
                            text=f"Verified claim {idx} of {total}...",
                        )

                    verified_results = verify_claims(
                        claims,
                        progress_callback=update_progress,
                    )
                    progress_bar.empty()

                    st.session_state["results"] = verified_results
                    st.session_state["checked_answer"] = answer_text

        except (ValueError, RuntimeError) as exc:
            st.error(f"Error: {exc}")
        except Exception as exc:
            st.error(f"An unexpected error occurred: {exc}")

# Display stored results (persists across reruns such as generating a corrected answer).
results = st.session_state.get("results")
if results is not None:
    if len(results) == 0:
        st.info(
            "No potentially outdated or time-sensitive factual claims were found "
            "in this answer."
        )
        if st.session_state.get("extract_model"):
            st.caption(f"Answered by: {st.session_state['extract_model']}")
    else:
        overall_score = round(sum(r["score"] for r in results) / len(results))
        overall_label, overall_color = get_overall_badge(overall_score)

        fresh_count = sum(1 for r in results if r["status"] == "fresh")
        uncertain_count = sum(1 for r in results if r["status"] == "uncertain")
        outdated_count = sum(1 for r in results if r["status"] == "outdated")
        error_count = sum(1 for r in results if r.get("error"))

        # Collect unique models that answered during extraction and verification.
        models_used: list[str] = []
        for model_name in [
            st.session_state.get("extract_model"),
            *(r.get("model") for r in results),
        ]:
            if model_name and model_name not in models_used:
                models_used.append(model_name)

        # Overall freshness summary card at the top
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 2, 1.5])
            with col1:
                st.metric("Overall Freshness Score", f"{overall_score} / 100")
            with col2:
                st.markdown(
                    f"**Claims Checked:** {len(results)}  \n"
                    f"Fresh: **{fresh_count}** • "
                    f"Uncertain: **{uncertain_count}** • "
                    f"Outdated: **{outdated_count}**"
                )
            with col3:
                st.badge(overall_label, color=overall_color)

            st.progress(overall_score / 100)
            if models_used:
                st.caption(f"Answered by: {', '.join(models_used)}")

            if error_count > 0:
                st.warning(
                    f"{error_count} claim(s) encountered a search or parsing error "
                    "and were marked as uncertain."
                )

        # Button to rewrite the original answer using verified current info
        if st.button("Generate corrected answer"):
            try:
                with st.spinner("Rewriting answer with verified current info..."):
                    corrected, corrected_model = generate_corrected_answer(
                        st.session_state["checked_answer"],
                        results,
                    )
                    st.session_state["corrected_answer"] = corrected
                    st.session_state["corrected_model"] = corrected_model
            except (ValueError, RuntimeError) as exc:
                st.error(f"Could not generate corrected answer: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error while rewriting answer: {exc}")

        if st.session_state.get("corrected_answer"):
            st.subheader("Corrected Answer")
            with st.container(border=True):
                st.markdown(escape_markdown(st.session_state["corrected_answer"]))
                corrected_model = st.session_state.get("corrected_model")
                if corrected_model:
                    st.caption(
                        f"Rewritten using today's verified web sources • Answered by: {corrected_model}"
                    )
                else:
                    st.caption("Rewritten using today's verified web sources.")

        st.subheader(f"Per-Claim Results ({len(results)})")
        for idx, item in enumerate(results, start=1):
            render_claim_card(idx, item)
