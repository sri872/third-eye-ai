"""Streamlit UI Component for NLP Incident Analysis.

Provides deterministic keyword and entity extraction, urgency scoring,
and contradiction detection for emergency caller transcripts.
"""

from typing import Dict, Any, Optional
import streamlit as st

from core.nlp.incident_analysis import analyze_incident


def render_nlp_tab(default_text: str = ""):
    """Render Section 3: NLP Incident Analysis."""
    st.subheader("3. Deterministic NLP Incident Analysis")
    st.markdown(
        "Rule-based entity extraction, incident classification, and urgency assessment — **No spaCy models, LLMs, or neural NLP.**"
    )

    # Transcript input area
    caller_text = st.text_area(
        "Caller Transcript / Emergency Description",
        value=default_text or "Two cars collided on GT Road near Phagwara sugar mill. One person is unconscious and bleeding, with a small petrol leak.",
        height=100,
    )

    if not caller_text.strip():
        st.warning("Please provide or select an incident description.")
        return None

    # Run deterministic analysis
    analysis = analyze_incident(caller_text)
    entities = analysis["entities"]

    # 1. Classification & Urgency Banner
    col_u1, col_u2 = st.columns([1, 2])
    with col_u1:
        urgency = analysis["urgency_level"]
        if urgency == "CRITICAL":
            st.error(f"🚨 **URGENCY: {urgency}**")
        elif urgency == "HIGH":
            st.warning(f"⚠️ **URGENCY: {urgency}**")
        elif urgency == "MEDIUM":
            st.info(f"ℹ️ **URGENCY: {urgency}**")
        else:
            st.success(f"🟢 **URGENCY: {urgency}**")
        st.caption(f"Incident Type: **{analysis['primary_incident_type']}**")

    with col_u2:
        st.markdown(f"**Urgency Rationale**: {analysis['urgency_reason']}")
        if analysis["has_contradictions"]:
            for contra in analysis["contradictions_detected"]:
                st.warning(contra)

    # 2. Extracted Entities Grid
    st.markdown("#### Extracted Entity Categories")
    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.markdown("**🚗 Vehicles**")
        if entities["vehicles"]:
            for v in entities["vehicles"]:
                col_str = f" ({v['color']})" if v.get("color") else ""
                st.write(f"- {v['count']} × {v['type']}{col_str}")
            st.caption(f"Total: {entities['total_vehicles_reported']} vehicle(s)")
        else:
            st.caption("No vehicles mentioned.")

    with e2:
        st.markdown("**📍 Locations**")
        if entities["locations"]:
            for loc in entities["locations"]:
                st.write(f"- {loc}")
        else:
            st.caption("No locations identified.")

    with e3:
        st.markdown("**🩹 Injuries**")
        if entities["injuries"]:
            for inj in entities["injuries"]:
                st.write(f"- {inj['keyword']} `({inj['category']})`")
        elif entities["has_negated_injuries"]:
            st.caption("Negated ('no injuries').")
        else:
            st.caption("None reported.")

    with e4:
        st.markdown("**🔥 Hazards**")
        if entities["hazards"]:
            for haz in entities["hazards"]:
                st.write(f"- {haz['keyword']} `({haz['category']})`")
        elif entities["has_negated_hazards"]:
            st.caption("Negated ('no hazards').")
        else:
            st.caption("None reported.")

    return analysis
