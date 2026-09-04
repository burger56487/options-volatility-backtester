"""Interactive dashboard (Streamlit) over the persisted run database."""

from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st


@st.cache_resource
def load_runs(db_path: str = "outputs/app.db") -> pd.DataFrame:
    connection = sqlite3.connect(db_path)
    return pd.read_sql_query(
        "SELECT run_id, created_at, status FROM runs "
        "ORDER BY created_at DESC",
        connection,
    )


def main() -> None:
    st.set_page_config(page_title="FICC Analytics", layout="wide")
    st.title("FICC Analytics Platform")
    runs = load_runs()
    st.dataframe(runs, use_container_width=True)
    selected = st.selectbox("Run ID", runs["run_id"])
    if selected:
        connection = sqlite3.connect("outputs/app.db")
        row = connection.execute(
            "SELECT metadata_json, metrics_json FROM runs WHERE run_id = ?",
            (selected,),
        ).fetchone()
        if row:
            st.json(row[0])
            st.json(row[1])


if __name__ == "__main__":
    main()
