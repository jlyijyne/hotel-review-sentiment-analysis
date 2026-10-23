"""Simple GUI for the NLP hotel-review coursework.
Place next to D1.csv and existing task scripts/results.
Run: py -m streamlit run hotel_review_gui.py
"""
from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Hotel Review NLP", page_icon="🏨", layout="wide")
st.title("🏨 Hotel Review Sentiment Analysis")
st.caption("London HotelRec · tasks 1–13 · interactive project dashboard")


def load_csv(name):
    path = ROOT / name
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {name}: {exc}")
        return None


def show_csv(name, rows=15):
    data = load_csv(name)
    if data is None:
        st.info(f"Missing file: {name}")
    else:
        st.caption(f"{name} · {len(data):,} rows")
        st.dataframe(data.head(rows), width="stretch")
    return data


def show_image(name):
    path = ROOT / name
    if path.is_file():
        st.image(str(path), caption=name, width="stretch")
    else:
        st.info(f"Missing image: {name}")


df = load_csv("D1.csv")
tabs = st.tabs([
    "Overview · 1–2", "Hotels · 3–4", "WordCloud & LDA · 5–6",
    "Empath & Ontology · 7–11", "Models & Bias · 12–13", "Run tasks"
])

with tabs[0]:
    st.subheader("Tasks 1–2 · SentiStrength & Pearson correlation")
    if df is None:
        st.warning("Put D1.csv in the same directory as this GUI.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Reviews", f"{len(df):,}")
        c2.metric("Hotels", str(df["hotel_id"].nunique()) if "hotel_id" in df else "—")
        if {"overall", "rating"}.issubset(df.columns):
            v = df[["overall", "rating"]].apply(pd.to_numeric, errors="coerce")
            r = v["overall"].corr(v["rating"])
            c3.metric("Pearson r", f"{r:.3f}" if pd.notna(r) else "—")
            st.write(
                "Average SentiStrength sentiment score for each user "
                "rating (1–5). Higher values indicate more positive reviews."
            )
            st.bar_chart(v.groupby("rating")["overall"].mean().rename("Mean SentiStrength overall"))
        else:
            c3.metric("Pearson r", "—")

        st.subheader("D1 Dataset – Hotel Reviews")
        selected_rating = st.selectbox(
            "Filter by user rating",
            ["All", 1, 2, 3, 4, 5],
            key="overview_rating_filter"
        )
        filtered = df.copy()
        if selected_rating != "All":
            filtered = filtered[
                pd.to_numeric(filtered["rating"], errors="coerce") == selected_rating
            ]
        st.caption(f"D1.csv · Showing {len(filtered):,} of {len(df):,} reviews")
        st.dataframe(filtered, width="stretch", height=500)

with tabs[1]:
    st.subheader("Tasks 3–4 · Hotel statistics and ambiguous subclasses")
    stats = load_csv("hotel_statistics_task4.csv")
    if stats is None:
        stats = load_csv("hotel_statistics.csv")
    if stats is None and df is not None and {"hotel_id", "rating"}.issubset(df.columns):
        data = df.copy()
        data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
        stats = data.groupby("hotel_id", as_index=False).agg(
            average=("rating", "mean"), std=("rating", "std"),
            reviews=("rating", "count")
        )
        epsilon = stats["std"].quantile(.75)
        stats["ambiguous"] = stats["std"] > epsilon
        st.caption(f"Statistics calculated from D1.csv; epsilon (75th percentile) = {epsilon:.3f}")
    if stats is not None:
        st.dataframe(stats.head(30), width="stretch")
        if "std" in stats:
            counts = stats["std"].dropna().value_counts(bins=12).sort_index()
            counts.index = counts.index.astype(str)
            st.bar_chart(counts)
        if "stars" in stats and "ambiguous" in stats:
            mask = stats.copy()
            mask["ambiguous"] = mask["ambiguous"].astype(str).str.lower().eq("true")
            mask["stars"] = pd.to_numeric(mask["stars"], errors="coerce")
            pct = mask.dropna(subset=["stars"]).groupby("stars")["ambiguous"].mean() * 100
            if not pct.empty:
                st.write("Percentage of ambiguous hotels by star class")
                st.bar_chart(pct)
        if "subclass" in stats:
            st.write("Ambiguous subclass distribution (hotels)")
            s = stats[stats["subclass"].isin(["Positive", "Negative"])]
            st.bar_chart(s["subclass"].value_counts())
    else:
        st.info("Run task 3 first to generate hotel_statistics.csv.")

with tabs[2]:
    st.subheader("Tasks 5–6 · WordCloud and LDA")
    left, right = st.columns(2)
    with left:
        show_image("wordcloud_positive.png")
    with right:
        show_image("wordcloud_negative.png")
    show_csv("lda_topics.csv", rows=5)
    dist = load_csv("lda_subclass_distribution.csv")
    if dist is not None:
        if "subclass" in dist:
            dist = dist.set_index("subclass")
        num = dist.select_dtypes(include="number")
        if not num.empty:
            st.write("Average topic distributions")
            st.bar_chart(num.T)
    else:
        st.info("Missing: lda_subclass_distribution.csv")

with tabs[3]:
    st.subheader("Tasks 7–11 · Empath, overlap and ontology")
    empath = load_csv("empath_results.csv")
    if empath is not None:
        subclass = st.selectbox("Empath subclass", ["Positive", "Negative"], key="empath_subclass")
        rows = empath[empath["subclass"] == subclass] if "subclass" in empath else empath
        st.dataframe(rows.sort_values("score", ascending=False).head(30) if "score" in rows else rows.head(30), width="stretch")
    else:
        st.info("Missing: empath_results.csv")
    show_csv("task9_agreement.csv", rows=10)
    show_image("ontology_histogram.png")
    st.caption("The ontology is implemented in task10.py; see the 'Run tasks' tab to execute scripts.")

with tabs[4]:
    st.subheader("Tasks 12–13 · Qwen vs SentiStrength and rating discrepancies")
    llm = load_csv("task12_results.csv")
    if llm is not None:
        st.write("LLM experiment records")
        st.dataframe(llm.head(30), width="stretch")
    else:
        st.info("Missing: task12_results.csv")
    bias = load_csv("task13_qwen_results.csv")
    if bias is None:
        bias = load_csv("task13_bias_cases.csv")
    if bias is not None:
        st.write("Strong rating–sentiment disagreements")
        st.metric("Flagged cases", len(bias))
        cols = [x for x in ["review_id", "rating", "overall", "qwen_label", "llm_label", "text"] if x in bias]
        st.dataframe(bias[cols].head(30) if cols else bias.head(30), width="stretch")

with tabs[5]:
    st.subheader("Run existing analysis scripts")
    st.write("Choose a task and point to its **existing** Python script. The app runs it locally and displays the console output. It does not automatically rerun any analysis when the dashboard opens.")
    filenames = {
        "1–2 SentiStrength & Pearson": "tasks_1_2_sentistrength.py",
        "3 Hotel statistics": "task3.py",
        "4 Ambiguous classes": "task4.py",
        "5 WordCloud": "task5.py",
        "6 LDA": "task6.py",
        "7 Empath": "task7.py",
        "8 Empath vs WordCloud": "task8.py",
        "9 Empath vs LDA": "task9.py",
        "10 Ontology dictionary": "task10.py",
        "11 Ontology histogram": "task11.py",
        "12 Qwen LLM (heavy)": "task12_llm.py",
        "13 Qwen bias check (heavy)": "task13_qwen.py",
    }
    task = st.selectbox("Task", list(filenames))
    filename = st.text_input("Script filename", value=filenames[task], key=task)
    script = (ROOT / filename).resolve()
    valid = script.is_file() and script.parent == ROOT and script.suffix.lower() == ".py"
    if not valid:
        st.warning("This script is not found here. Enter the real .py filename in your project folder.")
    if "heavy" in task:
        st.warning("This task loads a language model and may use several GB of RAM. Run it only when needed.")
    if st.button("Run selected task", type="primary", disabled=not valid):
        st.info(f"Executing: {script.name}")
        output = st.empty()
        lines = []
        try:
            with subprocess.Popen(
                [sys.executable, "-u", str(script)], cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1
            ) as process:
                for line in process.stdout:
                    lines.append(line.rstrip())
                    if len(lines) > 60:
                        lines.pop(0)
                    output.code("\n".join(lines), language="text")
                code = process.wait()
            if code == 0:
                st.success("Task finished. Refresh the page to view newly generated files.")
            else:
                st.error(f"Script exited with code {code}. Check the log above.")
        except Exception as exc:
            st.error(f"Could not run script: {exc}")

st.caption("This is a local coursework interface. Keep the dataset and Python scripts alongside this file.")
