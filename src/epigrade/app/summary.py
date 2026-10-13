"""The plain-language summary block, shared by every page that shows one."""

from __future__ import annotations

import streamlit as st

from epigrade.explain import groq_client
from epigrade.explain.explain import SOURCE_MODEL, explain, fact_hash, was_reviewed

LABEL = ("Generated from the computed numbers above by a language model, then checked against "
         "them. It adds no information of its own.")
TEMPLATE_LABEL = ("Written from the computed numbers above by a fixed template. It adds no "
                  "information of its own.")


def render(facts: dict, key: str) -> None:
    """Show the summary for `facts`. `key` must be unique on the page."""
    state_key = f"summary-{fact_hash(facts)}"
    try:
        result = st.session_state.get(state_key) or explain(facts)
    except Exception:  # noqa: BLE001 - a summary must never break the page it sits on
        st.caption("Plain-language summary unavailable.")
        return
    st.markdown("**Plain-language summary**")
    if result.source == SOURCE_MODEL:
        st.markdown(f":blue-badge[Language model: {result.model}] :green-badge[Passed the "
                    "number check]")
        st.caption(LABEL)
        if not was_reviewed(result.fact_hash):
            st.caption("This text has passed the automatic check only; it has not been "
                       "reviewed. Earlier reviews found errors of meaning the check misses.")
    else:
        st.markdown(":gray-badge[Deterministic template]")
        st.caption(TEMPLATE_LABEL)
    st.info(result.text)
    if groq_client.has_key() and st.button("Regenerate", key=f"regen-{key}"):
        st.session_state[state_key] = explain(facts, use_cache=False)
        st.rerun()
