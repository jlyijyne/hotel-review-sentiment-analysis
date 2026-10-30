"""Simple GUI for the NLP hotel-review coursework.
Place next to D1.csv and existing task scripts/results.
Run: py -m streamlit run hotel_review_gui.py
"""
from pathlib import Path
import ast
import re
from collections import Counter

import pandas as pd
import numpy as np
import altair as alt
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
        st.dataframe(data.head(rows), width="stretch", hide_index=True)
    return data


def show_image(name):
    path = ROOT / name
    if path.is_file():
        st.image(str(path), caption=name, width="stretch")
    else:
        st.info(f"Missing image: {name}")



@st.cache_data(show_spinner="Analyzing frequent WordCloud words …")
def get_common_wordcloud_words(file_path, file_mtime_ns):
    """Cache expensive WordCloud work; invalidate when the CSV changes."""
    from wordcloud import WordCloud, STOPWORDS
    data = pd.read_csv(file_path, usecols=["subclass", "text"])
    stopwords = set(STOPWORDS) | {
        "hotel", "room", "stay", "stayed", "one", "would", "could",
        "also", "really", "us", "get", "got"
    }
    result = {}
    for name in ["Positive", "Negative"]:
        corpus = " ".join(data.loc[data["subclass"] == name, "text"].dropna().astype(str))
        if corpus.strip():
            cloud = WordCloud(width=800, height=400, background_color="white",
                              stopwords=stopwords, max_words=200).generate(corpus)
            result[name] = set(w.lower() for w in cloud.words_)
    return result


@st.cache_data(show_spinner="Counting ontology vocabulary …")
def get_ontology_frequencies(file_path, file_mtime_ns, ontology_items):
    """Counts the original Task 11 words and computes size-normalized rates."""
    data = pd.read_csv(file_path, usecols=["subclass", "text"])
    count_rows = []
    for name in ["Positive", "Negative"]:
        text = " ".join(data.loc[data["subclass"] == name, "text"].dropna().astype(str)).lower()
        words_in_text = Counter(re.findall(r"\b[a-z]+\b", text))
        total_words = sum(words_in_text.values())
        for category, terms in ontology_items:
            count = 0
            for term in terms:
                term = term.lower().strip()
                if " " in term:
                    count += len(re.findall(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text))
                else:
                    count += words_in_text.get(term, 0)
            count_rows.append({
                "Subclass": name, "Category": category, "Count": count,
                "Per 1,000 words": 1000 * count / total_words if total_words else 0.0,
                "Total words": total_words,
            })
    return pd.DataFrame(count_rows)


def rating_sentiment(score):
    """Reference label based on the assignment's 1–5 user rating (3 is excluded)."""
    if pd.isna(score):
        return "Other"
    return "Positive" if score >= 4 else "Negative" if score <= 2 else "Other"


def sentiment_from_overall(score):
    if pd.isna(score):
        return "Invalid"
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"


def get_accuracy(truth, prediction):
    """Percentage of correctly predicted Positive/Negative rating classes."""
    valid = truth.isin(["Negative", "Positive"])
    truth = truth.loc[valid].astype(str)
    prediction = prediction.loc[valid].fillna("Invalid").astype(str)
    if len(truth) == 0:
        return None
    return 100 * (truth == prediction).sum() / len(truth)


def prepare_bias_cases(data, min_strength=2):
    """Select contradictory reviews directly from D1 using the chosen threshold."""
    required = {"rating", "overall"}
    if data is None or not required.issubset(data.columns):
        return pd.DataFrame()
    work = data.copy()
    work["rating"] = pd.to_numeric(work["rating"], errors="coerce")
    work["overall"] = pd.to_numeric(work["overall"], errors="coerce")
    high_neg = (work["rating"] >= 4) & (work["rating"] <= 5) & (work["overall"] <= -min_strength)
    low_pos = (work["rating"] >= 1) & (work["rating"] <= 2) & (work["overall"] >= min_strength)
    work = work.loc[high_neg | low_pos].copy()
    if work.empty:
        return work
    work["Case type"] = np.where(
        high_neg.loc[work.index], "Case A", "Case B"
    )
    work["Rating class"] = work["rating"].map(rating_sentiment)
    work["SentiStrength"] = work["overall"].map(sentiment_from_overall)
    return work


def attach_bias_qwen(cases, paths):
    """Match saved Qwen outputs by review_id, never by row order or a CSV's total size.

    Only saved predictions with a matching review ID are included.
    """
    cases = cases.copy()
    cases["Qwen"] = "Not evaluated"
    cases["Qwen source"] = ""
    if "review_id" not in cases.columns:
        return cases, []
    def normalized_id(values):
        return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    keys = normalized_id(cases["review_id"])
    assigned = pd.Series(False, index=cases.index)
    used_files = []
    for filename in paths:
        file_path = ROOT / filename
        if not file_path.is_file():
            continue
        results = load_csv(filename)
        if results is None or "review_id" not in results:
            continue
        prediction_col = next((name for name in ["llm_label", "qwen_label", "Qwen"]
                               if name in results.columns), None)
        if prediction_col is None:
            continue
        results = results.copy()
        results["_id"] = normalized_id(results["review_id"])
        raw = results[prediction_col].fillna("").astype(str).str.strip().str.capitalize()
        results["_qwen"] = raw
        # If a run was interrupted, blank predictions are not classified.
        results = results.loc[results["_qwen"].ne("") & results["_id"].notna()]
        if results.empty:
            continue
        results = results.drop_duplicates(subset="_id", keep="last")
        lookup = results.set_index("_id")["_qwen"]
        matched = keys.map(lookup)
        update = ~assigned & matched.notna()
        if update.any():
            cases.loc[update, "Qwen"] = matched.loc[update].values
            cases.loc[update, "Qwen source"] = filename
            assigned.loc[update] = True
            used_files.append(filename)
    return cases, used_files


df = load_csv("D1.csv")
tabs = st.tabs([
    "Overview · 1–2", "Hotels · 3–4", "WordCloud & LDA · 5–6",
    "Empath & Ontology · 7–11", "Models & Bias · 12–13"
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
            rating_means = (
                v.dropna(subset=["overall", "rating"])
                .groupby("rating", as_index=False)["overall"].mean()
            )
            rating_means["Rating"] = rating_means["rating"].map(lambda x: f"{x:g} / 5")
            sentiment_chart = alt.Chart(rating_means).mark_bar(size=55).encode(
                x=alt.X("Rating:N", title="User rating", sort=["1 / 5", "2 / 5", "3 / 5", "4 / 5", "5 / 5"],
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("overall:Q", title="Mean SentiStrength overall"),
                tooltip=["Rating:N", alt.Tooltip("overall:Q", title="Average overall", format=".3f")],
            ).properties(height=285)
            st.altair_chart(sentiment_chart, width="stretch")
            st.caption("Each bar is the mean SentiStrength overall score among reviews with that user rating; "
                       "the pattern illustrates the direction of the Pearson correlation.")
        else:
            c3.metric("Pearson r", "—")

        st.subheader("Task 1 · D1 dataset")
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
        st.caption(
            f"D1.csv · Showing {len(filtered):,} of {len(df):,} reviews. "
            "Each row is one review; rating is the user's score, while positive, negative "
            "and overall are SentiStrength outputs. Click a column header to sort."
        )
        st.dataframe(filtered.reset_index(drop=True), width="stretch", height=500, hide_index=True)

with tabs[1]:
    st.subheader("Tasks 3–4 · Hotel statistics and ambiguous subclasses")
    st.write(
        "Task 3 compares rating variation across hotels. A hotel is **Ambiguous** "
        "when its rating standard deviation exceeds ε (the 75th percentile). "
        "Task 4 splits these hotels into Positive and Negative subclasses "
        "according to the majority of users' ratings."
    )

    stats = load_csv("hotel_statistics_task4.csv")
    stats_source = "hotel_statistics_task4.csv"
    if stats is None:
        stats = load_csv("hotel_statistics.csv")
        stats_source = "hotel_statistics.csv"

    if stats is None and df is not None and {"hotel_id", "rating"}.issubset(df.columns):
        data = df.copy()
        data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
        stats = data.groupby("hotel_id", as_index=False).agg(
            average=("rating", "mean"),
            std=("rating", "std"),
            kurtosis=("rating", pd.Series.kurt),
            reviews=("rating", "count"),
        )
        star_col = next(
            (c for c in df.columns if c.strip().strip("*") == "hotel_stars"), None
        )
        if star_col:
            temp = df[["hotel_id", star_col]].copy()
            temp[star_col] = pd.to_numeric(temp[star_col], errors="coerce")
            star_map = temp.groupby("hotel_id")[star_col].first()
            stats["stars"] = stats["hotel_id"].map(star_map)
        stats_source = "D1.csv (calculated for display)"

    if stats is None:
        st.info("Run task 3 first or place hotel_statistics_task4.csv in the project folder.")
    else:
        stats = stats.copy()
        if "hotel_id" not in stats.columns and stats.index.name == "hotel_id":
            stats = stats.reset_index()
        if "hotel_id" not in stats.columns or "std" not in stats.columns:
            st.error("Hotel statistics must contain hotel_id and std columns.")
        else:
            for column in ["average", "std", "kurtosis", "reviews", "stars"]:
                if column in stats:
                    stats[column] = pd.to_numeric(stats[column], errors="coerce")
            # Star classes may have been added to D1 after task 3 was first run.
            if df is not None and "hotel_id" in df and "hotel_id" in stats:
                source_stars = next(
                    (c for c in df.columns if c.strip().strip("*") == "hotel_stars"), None
                )
                if source_stars:
                    star_data = df[["hotel_id", source_stars]].copy()
                    star_data[source_stars] = pd.to_numeric(star_data[source_stars], errors="coerce")
                    star_lookup = star_data.groupby("hotel_id")[source_stars].first()
                    from_d1 = stats["hotel_id"].map(star_lookup)
                    if "stars" not in stats:
                        stats["stars"] = from_d1
                    else:
                        stats["stars"] = stats["stars"].fillna(from_d1)
            valid_std = stats["std"].dropna()

            if valid_std.empty:
                st.warning("No standard deviations are available for this dataset.")
            else:
                # Same epsilon rule as task 3: 75th percentile, strict '>' threshold.
                epsilon = float(valid_std.quantile(0.75))
                if "ambiguous" in stats:
                    stats["ambiguous"] = (
                        stats["ambiguous"].astype(str).str.strip().str.lower().eq("true")
                    )
                else:
                    stats["ambiguous"] = stats["std"] > epsilon

                # Use hotel names when they are available in D1.
                if df is not None and {"hotel_id", "hotel_url"}.issubset(df.columns):
                    names = df[["hotel_id", "hotel_url"]].drop_duplicates("hotel_id").copy()
                    names["Hotel"] = (
                        names["hotel_url"].fillna("")
                        .str.split("-Reviews-", n=1).str[-1]
                        .str.split("-London", n=1).str[0]
                        .str.replace("_", " ", regex=False)
                    )
                    names = names.set_index(names["hotel_id"].astype(str))["Hotel"]
                    stats["Hotel"] = stats["hotel_id"].astype(str).map(names)

                # Recover subclasses from D1 when the saved task-4 statistics lack them.
                if "subclass" not in stats and df is not None and {"hotel_id", "subclass"}.issubset(df.columns):
                    classes = df.groupby("hotel_id")["subclass"].first()
                    stats["subclass"] = stats["hotel_id"].map(classes)
                if "subclass" not in stats and df is not None and {"hotel_id", "rating"}.issubset(df.columns):
                    rated = df[["hotel_id", "rating"]].copy()
                    rated["rating"] = pd.to_numeric(rated["rating"], errors="coerce")
                    pos = rated["rating"].ge(4).groupby(rated["hotel_id"]).sum()
                    neg = rated["rating"].le(2).groupby(rated["hotel_id"]).sum()
                    diff = stats["hotel_id"].map(pos).fillna(0) - stats["hotel_id"].map(neg).fillna(0)
                    stats["subclass"] = "Not Ambiguous"
                    stats.loc[stats["ambiguous"] & diff.gt(0), "subclass"] = "Positive"
                    stats.loc[stats["ambiguous"] & diff.lt(0), "subclass"] = "Negative"
                    stats.loc[stats["ambiguous"] & diff.eq(0), "subclass"] = "Undetermined"

                ambiguous_stats = stats[stats["ambiguous"]].copy()
                c1, c2, c3 = st.columns(3)
                c1.metric("Hotels", f"{len(stats):,}")
                c2.metric("ε · 75th percentile", f"{epsilon:.3f}")
                c3.metric("Ambiguous hotels", f"{len(ambiguous_stats):,} ({len(ambiguous_stats)/len(stats)*100:.1f}%)")
                st.caption(
                    f"Source: {stats_source}. Ambiguous means rating std > {epsilon:.3f}; "
                    "ties at the threshold are not included."
                )

                st.markdown("#### Task 3 · Hotel statistics")
                left, right = st.columns(2)
                with left:
                    order = st.selectbox("Sort hotels", ["Highest std first", "Lowest std first", "Hotel ID"], key="hotels_sort")
                with right:
                    selection = st.selectbox("Show hotels", ["All", "Ambiguous only", "Not ambiguous"], key="hotels_filter")
                visible = stats.copy()
                if selection == "Ambiguous only":
                    visible = visible[visible["ambiguous"]]
                elif selection == "Not ambiguous":
                    visible = visible[~visible["ambiguous"]]
                if order == "Highest std first":
                    visible = visible.sort_values("std", ascending=False)
                elif order == "Lowest std first":
                    visible = visible.sort_values("std")
                else:
                    visible = visible.sort_values("hotel_id")
                st.caption(f"Showing all {len(visible):,} matching hotels. Scroll or sort the table to inspect them.")
                visible_display = visible.copy()
                for col in ["average", "std", "kurtosis"]:
                    if col in visible_display:
                        visible_display[col] = visible_display[col].round(3)
                st.dataframe(visible_display, width="stretch", height=390, hide_index=True)

                st.markdown("#### Task 3 · Lowest and highest standard deviations")
                st.write("A low std means ratings are similar; a high std means reviewers disagree more.")
                first, second = st.columns(2)
                display_columns = [c for c in ["Hotel", "hotel_id", "average", "std", "kurtosis", "stars", "reviews"] if c in stats]
                with first:
                    st.markdown("**Five lowest std hotels**")
                    st.dataframe(stats.nsmallest(5, "std")[display_columns].round(3), width="stretch", hide_index=True)
                with second:
                    st.markdown("**Five highest std hotels**")
                    st.dataframe(stats.nlargest(5, "std")[display_columns].round(3), width="stretch", hide_index=True)

                st.markdown("#### Task 3 · Standard deviation distribution")
                st.write("Each bar counts hotels in a standard-deviation range. The orange line marks ε.")
                # At most approximately seven bins with rounded, easy-to-read tick marks.
                low, high = float(valid_std.min()), float(valid_std.max())
                step = next((s for s in [0.1, 0.2, 0.5, 1.0, 2.0] if (high - low) / s <= 6.8), 2.0)
                start = np.floor(low / step) * step
                finish = (np.floor(high / step) + 1) * step
                edges = np.arange(start, finish + step / 2, step)
                hist, _ = np.histogram(valid_std, bins=edges)
                hist_data = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "hotels": hist})
                ticks = [round(float(x), 2) for x in edges]
                bars = alt.Chart(hist_data).mark_bar(opacity=0.85, stroke="white", strokeWidth=0.5).encode(
                    x=alt.X("start:Q", title="Standard deviation (rating points)",
                            axis=alt.Axis(values=ticks, labelAngle=0, format=".2~f")),
                    x2="end:Q",
                    y=alt.Y("hotels:Q", title="Number of hotels", axis=alt.Axis(tickMinStep=1)),
                    tooltip=[alt.Tooltip("start:Q", title="From", format=".2f"),
                             alt.Tooltip("end:Q", title="To", format=".2f"),
                             alt.Tooltip("hotels:Q", title="Hotels")],
                )
                cutoff = alt.Chart(pd.DataFrame({"epsilon": [epsilon]})).mark_rule(
                    color="#ea8a2f", strokeWidth=3
                ).encode(x="epsilon:Q")
                st.altair_chart(bars + cutoff, width="stretch")

                st.markdown("#### Task 3 · Ambiguous hotels by star class")
                if "stars" in stats and stats["stars"].notna().any():
                    starred = stats.dropna(subset=["stars", "std"]).copy()
                    # Keep the original star ratings as separate classes,
                    # including half-star values such as 2.5 or 3.5.
                    summary = starred.groupby("stars").agg(
                        hotels=("hotel_id", "count"),
                        ambiguous_hotels=("ambiguous", "sum"),
                        mean_std=("std", "mean"),
                    ).reset_index()
                    summary["ambiguous_pct"] = (
                        summary["ambiguous_hotels"] / summary["hotels"] * 100
                    )
                    summary["star_label"] = summary["stars"].map(lambda x: f"{x:g} ★")
                    pct_chart = alt.Chart(summary).mark_bar().encode(
                        x=alt.X("star_label:N", sort=summary["star_label"].tolist(),
                                title="Hotel star class", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("ambiguous_pct:Q", title="Ambiguous hotels (%)",
                                scale=alt.Scale(domain=[0, 100])),
                        tooltip=[alt.Tooltip("star_label:N", title="Star class"),
                                 alt.Tooltip("hotels:Q", title="All hotels"),
                                 alt.Tooltip("ambiguous_hotels:Q", title="Ambiguous"),
                                 alt.Tooltip("ambiguous_pct:Q", title="Ambiguous %", format=".1f")],
                    ).properties(height=280)
                    st.altair_chart(pct_chart, width="stretch")
                    st.dataframe(
                        summary[["star_label", "hotels", "ambiguous_hotels", "mean_std", "ambiguous_pct"]]
                        .rename(columns={"star_label": "Stars", "hotels": "Hotels",
                                         "ambiguous_hotels": "Ambiguous hotels",
                                         "mean_std": "Average std", "ambiguous_pct": "Ambiguous %"})
                        .round(2),
                        width="stretch", hide_index=True,
                    )
                    st.caption(
                        f"{len(stats) - len(starred)} hotels lack a usable star class "
                        "and are excluded from this chart."
                    )
                else:
                    st.info("No hotel star classes found. Add hotel_stars to D1 or the statistics file.")

                st.markdown("#### Task 4 · Ambiguous hotel subclasses")
                if "subclass" in stats:
                    classes = ambiguous_stats["subclass"].fillna("Undetermined").astype(str)
                    class_counts = classes.value_counts().reindex(["Positive", "Negative", "Undetermined"], fill_value=0)
                    class_counts = class_counts[class_counts > 0]
                    if not class_counts.empty:
                        subclass_chart = pd.DataFrame({
                            "Subclass": class_counts.index,
                            "Hotels": class_counts.values,
                            "Percentage": class_counts.values / len(ambiguous_stats) * 100,
                        })
                        st.write("Percentages are calculated out of **all Ambiguous hotels**, including ties if any.")
                        class_bars = alt.Chart(subclass_chart).mark_bar().encode(
                            x=alt.X("Subclass:N", sort=["Positive", "Negative", "Undetermined"],
                                    axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("Percentage:Q", title="Share of ambiguous hotels (%)", scale=alt.Scale(domain=[0, 100])),
                            tooltip=["Subclass:N", "Hotels:Q", alt.Tooltip("Percentage:Q", format=".1f")],
                        ).properties(height=280)
                        st.altair_chart(class_bars, width="stretch")
                        st.dataframe(subclass_chart.round(1), width="stretch", hide_index=True)
                    else:
                        st.info("No Ambiguous hotels with subclass labels were found.")
                else:
                    st.info("Task 4 subclass labels are missing. Run task 4 to classify Ambiguous hotels.")

with tabs[2]:
    st.subheader("Tasks 5–6 · WordCloud and LDA")
    st.write(
        "**Task 5:** WordClouds summarize frequent words in all reviews of hotels labeled "
        "Positive or Negative. These are **hotel-level subclasses**: a Positive hotel can "
        "still have negative individual reviews. Larger words appear more often, not "
        "necessarily more positively or negatively."
    )

    st.markdown("#### Task 5 · WordClouds")
    left, right = st.columns(2)
    with left:
        st.markdown("**Positive subclass**")
        show_image("wordcloud_positive.png")
    with right:
        st.markdown("**Negative subclass**")
        show_image("wordcloud_negative.png")

    st.markdown("#### Task 6 · Five LDA topics")
    st.write(
        "LDA groups words that frequently occur together. Each topic is represented by "
        "its five highest-weighted words. A topic does **not** automatically indicate sentiment."
    )
    topics = load_csv("lda_topics.csv")
    parsed_topics = {}
    if topics is not None and {"topic", "words"}.issubset(topics.columns):
        for _, topic_row in topics.iterrows():
            try:
                topic_num = int(topic_row["topic"])
                words = [w.strip().lower() for w in str(topic_row["words"]).split(",") if w.strip()]
                if words:
                    parsed_topics[topic_num] = words[:5]
            except (TypeError, ValueError):
                pass
        topic_rows = pd.DataFrame([
            {"Topic": f"Topic {i}", "Five most important words": ", ".join(parsed_topics[i])}
            for i in sorted(parsed_topics)
        ])
        if len(topic_rows) != 5:
            st.warning("Expected five topics in lda_topics.csv; check the file if some are missing.")
        st.dataframe(topic_rows, width="stretch", hide_index=True, height=245)
    elif topics is None:
        st.info("Missing lda_topics.csv: run Task 6 to display the five topic word lists.")
    else:
        st.warning("lda_topics.csv should have topic and words columns.")

    st.markdown("#### Task 6 · Average topic distribution by subclass")
    st.write(
        "The bars show the **average share** of each topic in the reviews of a subclass. "
        "For each subclass, the five average shares add up to approximately 100%."
    )
    dist = load_csv("lda_subclass_distribution.csv")
    if dist is not None:
        if "subclass" not in dist.columns and len(dist.columns) > 0:
            # Sometimes the index is saved with an unnamed or first column.
            if str(dist.columns[0]).startswith("Unnamed"):
                dist = dist.rename(columns={dist.columns[0]: "subclass"})
        if "subclass" in dist.columns:
            topic_cols = [f"lda_topic_{i}" for i in range(1, 6)]
            if set(topic_cols).issubset(dist.columns):
                plot_dist = dist[dist["subclass"].isin(["Negative", "Positive"])].copy()
                for col in topic_cols:
                    plot_dist[col] = pd.to_numeric(plot_dist[col], errors="coerce")
                chart_data = plot_dist.melt(
                    id_vars="subclass", value_vars=topic_cols,
                    var_name="topic", value_name="proportion"
                )
                chart_data["topic_label"] = chart_data["topic"].str.replace(
                    "lda_topic_", "Topic ", regex=False
                )
                grouped_bars = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("topic_label:N", title="LDA topic",
                            sort=[f"Topic {i}" for i in range(1, 6)],
                            axis=alt.Axis(labelAngle=0)),
                    xOffset=alt.XOffset("subclass:N", sort=["Negative", "Positive"]),
                    y=alt.Y("proportion:Q", title="Average topic proportion",
                            axis=alt.Axis(format=".0%")),
                    color=alt.Color("subclass:N", title="Subclass",
                                    sort=["Negative", "Positive"]),
                    tooltip=["subclass:N", "topic_label:N",
                             alt.Tooltip("proportion:Q", format=".1%", title="Topic share")],
                ).properties(height=360)
                st.altair_chart(grouped_bars, width="stretch")

                display_dist = plot_dist.set_index("subclass")[topic_cols].T * 100
                display_dist.index = [f"Topic {i}" for i in range(1, 6)]
                display_dist.index.name = "Topic"
                st.caption("The table below reports the same averages as percentages; it is not a count of reviews.")
                st.dataframe(
                    display_dist.reset_index().round(1),
                    width="stretch", hide_index=True
                )
            else:
                st.warning("lda_subclass_distribution.csv is missing one or more lda_topic_1 ... lda_topic_5 columns.")
        else:
            st.warning("lda_subclass_distribution.csv needs a subclass column.")
    else:
        st.info("Missing lda_subclass_distribution.csv: run Task 6 to plot topic distributions.")

    st.markdown("#### Task 6 · WordCloud versus LDA")
    st.write(
        "For each subclass, this comparison takes its **20 most frequent WordCloud words** "
        "and matches them against the five most important words from **all five LDA topics** "
        "(up to 25, with duplicates counted only once). Matching means the exact same "
        "word occurs in both sets. Semantically related but different words are not counted."
    )
    if not parsed_topics:
        st.info("The LDA topic words are needed for this comparison.")
    else:
        review_source = "D1.csv"
        review_data = df
        if review_data is None or not {"text", "subclass"}.issubset(review_data.columns):
            for candidate in ["D1_task6.csv", "D1_task4.csv"]:
                candidate_df = load_csv(candidate) if (ROOT / candidate).is_file() else None
                if candidate_df is not None and {"text", "subclass"}.issubset(candidate_df.columns):
                    review_data = candidate_df
                    review_source = candidate
                    break
        if review_data is None or not {"text", "subclass"}.issubset(review_data.columns):
            st.info(
                "WordCloud overlap requires text and subclass columns. "
                "Save the Task 4 labels in D1.csv or D1_task4.csv."
            )
        else:
            try:
                from wordcloud import WordCloud, STOPWORDS
            except ImportError:
                st.info("Install WordCloud to calculate the overlap: py -m pip install wordcloud")
            else:
                # Same stopwords as in the Task 5 code provided for the WordCloud images.
                extra_stopwords = {
                    "hotel", "room", "stay", "stayed", "one", "would", "could",
                    "also", "really", "us", "get", "got"
                }
                stopwords = set(STOPWORDS) | extra_stopwords
                lda_words = set().union(*(set(words) for words in parsed_topics.values()))
                overlap_rows = []
                word_lists = {}
                for subclass in ["Positive", "Negative"]:
                    corpus = " ".join(
                        review_data.loc[review_data["subclass"] == subclass, "text"]
                        .dropna().astype(str)
                    )
                    if not corpus.strip():
                        st.warning(f"No review texts in D1.csv for {subclass} subclass.")
                        continue
                    cloud = WordCloud(
                        width=800, height=400, background_color="white",
                        stopwords=stopwords, max_words=20
                    ).generate(corpus)
                    common_words = list(cloud.words_.keys())
                    matches = sorted(set(common_words) & lda_words)
                    word_lists[subclass] = (common_words, matches)
                    overlap_rows.append({
                        "Subclass": subclass,
                        "Top WordCloud words": len(common_words),
                        "Matching words": len(matches),
                        "Overlap (%)": round(100 * len(matches) / len(common_words), 1) if common_words else 0,
                        "Exact overlapping words": ", ".join(matches) if matches else "—",
                    })
                if overlap_rows:
                    st.dataframe(pd.DataFrame(overlap_rows), width="stretch", hide_index=True)
                    for subclass, (common_words, matches) in word_lists.items():
                        with st.expander(f"{subclass}: 20 WordCloud words and overlap details"):
                            st.write("**Most frequent WordCloud words:** " + ", ".join(common_words))
                            st.write("**Also in LDA:** " + (", ".join(matches) if matches else "None"))
                    st.caption(
                        "Computed from D1 using the Task 5 stopword settings. If the saved WordCloud "
                        "images were generated with different settings, their displayed words can differ. "
                        "An exact match means that the same word appears in both methods."
                    )

    st.markdown("#### Task 6 · Topic scores in D1")
    lda_columns = [f"lda_topic_{i}" for i in range(1, 6)]
    if df is not None and set(lda_columns).issubset(df.columns):
        scores_data = df.copy()
        scores_file = "D1.csv"
    elif (ROOT / "D1_task6.csv").is_file():
        saved_scores = load_csv("D1_task6.csv")
        if saved_scores is not None and set(lda_columns).issubset(saved_scores.columns):
            if (df is not None and "review_id" in df.columns and
                "review_id" in saved_scores.columns):
                scores_data = df.copy()
                id_base = scores_data["review_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
                saved_scores = saved_scores.copy()
                saved_scores["_join_id"] = (
                    saved_scores["review_id"].astype("string").str.strip()
                    .str.replace(r"\.0$", "", regex=True)
                )
                saved_scores = saved_scores.drop_duplicates("_join_id", keep="last").set_index("_join_id")
                for column in lda_columns + (["lda_main_topic"] if "lda_main_topic" in saved_scores else []):
                    scores_data[column] = id_base.map(saved_scores[column])
                scores_file = "D1.csv + D1_task6.csv"
            else:
                scores_data = saved_scores
                scores_file = "D1_task6.csv"
        else:
            scores_data = df if df is not None else None
            scores_file = "D1.csv"
    else:
        scores_data = df if df is not None else None
        scores_file = "D1.csv"

    if scores_data is not None and set(lda_columns).issubset(scores_data.columns):
        show_columns = [c for c in ["review_id", "hotel_id", "subclass", "text", *lda_columns, "lda_main_topic"]
                        if c in scores_data.columns]
        topic_scores = scores_data[lda_columns].apply(pd.to_numeric, errors="coerce")
        has_topic = topic_scores.notna().any(axis=1) & topic_scores.fillna(0).abs().sum(axis=1).gt(0)
        if "hotel_id" in scores_data.columns:
            hotel_has_topic = has_topic.groupby(scores_data["hotel_id"]).transform("any")
        else:
            hotel_has_topic = has_topic
        st.write(
            "Each review has a share for each of the five topics and a dominant topic. "
            "These are **per-review** scores, unlike the subclass averages above."
        )
        topic_filter = st.selectbox(
            "Filter topic scores",
            ["All reviews", "Reviews with topic scores", "Reviews without topic scores",
             "Hotels without any topic scores"],
            key="lda_topic_filter",
        )
        if topic_filter == "Reviews with topic scores":
            subset_scores = scores_data.loc[has_topic]
        elif topic_filter == "Reviews without topic scores":
            subset_scores = scores_data.loc[~has_topic]
        elif topic_filter == "Hotels without any topic scores":
            subset_scores = scores_data.loc[~hotel_has_topic]
        else:
            subset_scores = scores_data
        st.caption(f"Source: {scores_file} · Showing {len(subset_scores):,} of {len(scores_data):,} reviews.")
        st.dataframe(subset_scores[show_columns].reset_index(drop=True),
                     width="stretch", height=365, hide_index=True)
    else:
        st.warning(
            "No per-review lda_topic_1 ... lda_topic_5 columns were found. "
            "Run your task-6 script and keep D1_task6.csv."
        )

with tabs[3]:
    st.subheader("Tasks 7–11 · Empath, agreement and hotel ontology")
    st.write(
        "**Tasks 7–9** compare Empath's thematic word categories with the common review "
        "vocabulary and LDA topics. **Tasks 10–11** group hotel-related words into aspects "
        "and count their mentions in Positive and Negative hotel subclasses."
    )

    # Use the most complete review dataset available for on-page calculations.
    review_source = "D1.csv"
    review_data = df
    if review_data is None or not {"subclass", "text"}.issubset(review_data.columns):
        for candidate in ["D1_task6.csv", "D1_task4.csv"]:
            candidate_data = load_csv(candidate) if (ROOT / candidate).is_file() else None
            if candidate_data is not None and {"subclass", "text"}.issubset(candidate_data.columns):
                review_data = candidate_data
                review_source = candidate
                break
    have_reviews = review_data is not None and {"subclass", "text"}.issubset(review_data.columns)
    if not have_reviews:
        st.warning("Tasks 8 and 11 need a dataset with 'text' and 'subclass' columns (D1.csv or D1_task4.csv).")

    # TASK 7: Empath's nonzero categories for both subclasses.
    st.markdown("### Task 7 · Empath categories")
    st.write(
        "Empath identifies thematic vocabulary categories such as *hygiene*, *sleep* and "
        "*restaurant*. Here the scores come from each subclass's **combined review text**. "
        "Only categories with a score greater than zero are included. Scores are normalized "
        "word-category frequencies."
    )
    empath = load_csv("empath_results.csv")
    active_empath = None
    if empath is not None and {"subclass", "category", "score"}.issubset(empath.columns):
        empath = empath.copy()
        empath["score"] = pd.to_numeric(empath["score"], errors="coerce")
        active_empath = empath[
            empath["subclass"].isin(["Positive", "Negative"]) &
            (empath["score"] > 0)
        ].copy()
        summary = (active_empath.groupby("subclass")["category"]
                   .nunique().reindex(["Positive", "Negative"], fill_value=0))
        pcol, ncol = st.columns(2)
        pcol.metric("Active Empath categories · Positive", int(summary["Positive"]))
        ncol.metric("Active Empath categories · Negative", int(summary["Negative"]))
        selected_empath = st.selectbox("Show categories for", ["Positive", "Negative"], key="empath_subclass")
        selected_rows = (active_empath[active_empath["subclass"] == selected_empath]
                         .sort_values("score", ascending=False))
        st.caption(f"All {len(selected_rows)} non-zero categories are shown below; click a column header to sort.")
        st.dataframe(selected_rows.reset_index(drop=True), width="stretch", height=360, hide_index=True)
        st.write("**15 highest-scoring categories**")
        chart_emp = selected_rows.head(15).sort_values("score")
        if not chart_emp.empty:
            emp_chart = alt.Chart(chart_emp).mark_bar().encode(
                x=alt.X("score:Q", title="Normalized category score"),
                y=alt.Y("category:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
                tooltip=["category:N", alt.Tooltip("score:Q", format=".4f")],
            ).properties(height=390)
            st.altair_chart(emp_chart, width="stretch")

    else:
        st.warning("empath_results.csv is missing, or it lacks subclass, category or score columns.")

    # TASK 8: Compare Empath categories with the frequent words in the common corpus.
    st.markdown("### Task 8 · Empath versus common review vocabulary")
    st.write(
        "The original Task 8 compared **Empath category names** with frequent WordCloud words. "
        "This is a strict string match: *eating* and *breakfast*, for example, do not match. "
        "We also show an optional **lexicon-based comparison**, which checks the words "
        "belonging to each Empath category. In both calculations the denominator is the "
        "number of active Empath categories in that subclass."
    )
    if active_empath is not None and have_reviews:
        try:
            from wordcloud import WordCloud, STOPWORDS
        except ImportError:
            st.info("Install WordCloud to calculate Task 8: py -m pip install wordcloud")
        else:
            task8_rows = []
            wc_data = {}
            review_path = ROOT / review_source
            wordcloud_vocab = get_common_wordcloud_words(str(review_path), review_path.stat().st_mtime_ns)
            for name in ["Positive", "Negative"]:
                cats = set(active_empath.loc[
                    active_empath["subclass"] == name, "category"].astype(str).str.lower())
                wc_words = wordcloud_vocab.get(name, set())
                if not wc_words or not cats:
                    st.warning(f"Cannot calculate Task 8 for {name}: review text or Empath categories are missing.")
                    continue
                matches = sorted(cats & wc_words)
                wc_data[name] = (wc_words, cats)
                task8_rows.append({
                    "Subclass": name,
                    "Empath categories": len(cats),
                    "Exact name matches": len(matches),
                    "Name overlap (%)": 100 * len(matches) / len(cats),
                    "Matching names": ", ".join(matches) if matches else "—",
                })
            if task8_rows:
                st.dataframe(pd.DataFrame(task8_rows).round(2), width="stretch", hide_index=True)
                st.caption(
                    "Exact-name overlap = categories whose names occur among the 200 most "
                    "frequent WordCloud words / all non-zero Empath categories. "
                    "An exact name match is a very conservative indicator of agreement."
                )
                try:
                    from empath import Empath
                    lexicon = Empath()
                except (ImportError, OSError, ValueError) as exc:
                    st.info(f"Install/configure Empath to enable category-vocabulary comparisons: {exc}")
                else:
                    lexicon_rows = []
                    for name, (wc_words, cats) in wc_data.items():
                        matched = sorted(
                            cat for cat in cats
                            if wc_words.intersection(
                                str(w).lower() for w in lexicon.cats.get(cat, [])
                            )
                        )
                        lexicon_rows.append({
                            "Subclass": name,
                            "Empath categories": len(cats),
                            "Lexicon matches": len(matched),
                            "Lexicon overlap (%)": 100 * len(matched) / len(cats),
                            "Matching categories": ", ".join(matched) if matched else "—",
                        })
                    st.write("**Additional context-aware vocabulary coverage (optional)**")
                    st.dataframe(pd.DataFrame(lexicon_rows).round(2), width="stretch", hide_index=True)
                    st.caption(
                        "Here a category matches when at least one of its own Empath vocabulary "
                        "terms appears among the WordCloud terms. This is a different, broader "
                        "measure than Task 8's original name-matching code."
                    )
    else:
        st.info("Task 8 requires empath_results.csv plus review text and subclass labels.")

    # TASK 9: Compare category vocabulary with the two leading LDA topics per subclass.
    st.markdown("### Task 9 · Empath versus LDA")
    st.write(
        "The analysis uses the **two highest-proportion LDA topics in each subclass**. "
        "An Empath category matches if at least one word from those topics occurs in its "
        "category vocabulary."
    )
    agreement = load_csv("task9_agreement.csv")
    if agreement is not None and {"subclass", "empath_categories", "matching_categories", "agreement_ratio"}.issubset(agreement.columns):
        agreement = agreement.copy()
        agreement["agreement_ratio"] = pd.to_numeric(agreement["agreement_ratio"], errors="coerce")
        agreement["Agreement (%)"] = agreement["agreement_ratio"] * 100
        view = agreement[["subclass", "empath_categories", "matching_categories", "Agreement (%)"]].copy()
        view.columns = ["Subclass", "Empath categories", "Matched categories", "Agreement (%)"]
        st.dataframe(view.round(2), width="stretch", hide_index=True)
        agree_chart = alt.Chart(view).mark_bar().encode(
            x=alt.X("Subclass:N", title="Subclass", sort=["Positive", "Negative"],
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Agreement (%):Q", title="Matching Empath categories (%)",
                    scale=alt.Scale(domain=[0, max(10.0, float(view["Agreement (%)"].max()) * 1.15)])),
            color=alt.Color("Subclass:N", sort=["Positive", "Negative"]),
            tooltip=["Subclass:N", "Matched categories:Q", "Empath categories:Q",
                     alt.Tooltip("Agreement (%):Q", format=".2f")],
        ).properties(height=300)
        st.altair_chart(agree_chart, width="stretch")
        st.caption(
            "A higher value means the selected LDA words touch more Empath category "
            "vocabularies. Because the choice of topics affects the result, this is not "
            "a sentiment-classification performance metric."
        )
        # Show matching category names, in addition to the percentages saved by Task 9.
        if active_empath is not None:
            try:
                from empath import Empath
                lexicon9 = Empath()
                topics9 = load_csv("lda_topics.csv")
                distribution9 = load_csv("lda_subclass_distribution.csv")
                if (topics9 is not None and distribution9 is not None and
                    {"topic", "words"}.issubset(topics9.columns)):
                    if "subclass" not in distribution9 and str(distribution9.columns[0]).startswith("Unnamed"):
                        distribution9 = distribution9.rename(columns={distribution9.columns[0]: "subclass"})
                    details = []
                    for name in ["Positive", "Negative"]:
                        subset = distribution9.loc[distribution9["subclass"] == name]
                        if subset.empty:
                            continue
                        topic_shares = pd.to_numeric(subset.iloc[0].drop("subclass"), errors="coerce")
                        selected_topic_names = topic_shares.nlargest(2).index
                        target_words = set()
                        for topic_col in selected_topic_names:
                            topic_id = int(str(topic_col).split("_")[-1])
                            series = topics9.loc[pd.to_numeric(topics9["topic"], errors="coerce") == topic_id, "words"]
                            if not series.empty:
                                target_words.update(w.strip().lower() for w in str(series.iloc[0]).split(",") if w.strip())
                        categories = active_empath.loc[active_empath["subclass"] == name, "category"].astype(str)
                        matched = sorted(cat for cat in categories if
                                         target_words.intersection(
                                             str(word).lower() for word in lexicon9.cats.get(cat, [])))
                        details.append({"Subclass": name,
                                        "Selected topics": ", ".join(str(x).replace("lda_topic_", "Topic ") for x in selected_topic_names),
                                        "Matching categories": ", ".join(matched) if matched else "—"})
                    if details:
                        with st.expander("Show selected LDA topics and matching Empath category names"):
                            st.dataframe(pd.DataFrame(details), width="stretch", hide_index=True)
            except (ImportError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                st.caption(f"Detailed matched categories could not be calculated: {exc}")
    else:
        st.info("Missing task9_agreement.csv or required columns; run Task 9 to show its saved results.")

    # TASK 10: Show the actual ontology (loaded as a literal from task10.py).
    st.markdown("### Task 10 · Hotel-review ontology")
    st.write(
        "Our manually constructed ontology groups review vocabulary into **five** hotel "
        "aspects: Price, Service, Room, Location and Food. "
        "The vocabulary is a simple dictionary of related words."
    )
    fallback = {
        "Price": ["price", "cost", "cheap", "expensive", "affordable", "value", "budget", "overpriced", "reasonable", "money"],
        "Service": ["service", "staff", "friendly", "helpful", "rude", "reception", "receptionist", "customer", "polite", "manager"],
        "Room": ["room", "bedroom", "bed", "bathroom", "shower", "clean", "dirty", "comfortable", "pillow", "towel"],
        "Location": ["location", "central", "nearby", "distance", "station", "transport", "accessible", "metro", "underground", "walking"],
        "Food": ["food", "breakfast", "dinner", "lunch", "restaurant", "meal", "delicious", "buffet", "menu", "coffee"],
    }
    ontology = dict(fallback)
    ontology_source = "built-in vocabulary (task10.py not found)"
    ontology_path = ROOT / "task10.py"
    if ontology_path.is_file():
        try:
            source_tree = ast.parse(ontology_path.read_text(encoding="utf-8-sig"))
            definitions = [node.value for node in source_tree.body
                           if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "ontology" for t in node.targets)]
            if definitions:
                loaded = ast.literal_eval(definitions[-1])
                if isinstance(loaded, dict) and all(isinstance(v, (list, tuple)) for v in loaded.values()):
                    ontology = {str(k): [str(w) for w in v] for k, v in loaded.items()}
                    ontology_source = "task10.py"
            else:
                st.warning("No literal 'ontology = {...}' dictionary found in task10.py; using the built-in list.")
        except (OSError, SyntaxError, ValueError, TypeError) as exc:
            st.warning(f"Could not read task10.py ontology ({exc}); using the built-in list.")
    # The coursework implementation intentionally uses five classes (Parking is excluded).
    # Keep the same five even if task10.py contains additional categories.
    ontology = {category: ontology.get(category, fallback[category]) for category in fallback}
    st.caption("The categories use manually selected related words; a word may occur in more than one aspect.")
    display_ontology = pd.DataFrame([
        {"Aspect": category, "Terms": ", ".join(terms), "Number of terms": len(terms)}
        for category, terms in ontology.items()
    ])
    st.dataframe(display_ontology, width="stretch", hide_index=True)
    if ontology_source != "task10.py":
        updated_source = "# Hotel-review ontology: five selected categories\nontology = " + repr(ontology) + "\n"
        st.download_button("Download the five-category ontology", updated_source,
                           file_name="task10_updated.py", mime="text/x-python")

    # TASK 11: Recompute the histogram from the displayed ontology and review text.
    st.markdown("### Task 11 · Frequency of ontology categories")
    st.write(
        "The bars count occurrences of the words in each ontology aspect. Choose "
        "raw word counts to reproduce the original analysis, or **per 1,000 words** to "
        "compare subclasses fairly when they contain different amounts of review text."
    )
    if have_reviews:
        review_path = ROOT / review_source
        ontology_items = tuple((str(category), tuple(str(w) for w in words))
                               for category, words in ontology.items())
        ontology_counts = get_ontology_frequencies(
            str(review_path), review_path.stat().st_mtime_ns, ontology_items
        )
        if not ontology_counts.empty:
            metric = st.radio("Show", ["Raw mentions", "Mentions per 1,000 words"],
                              horizontal=True, key="ontology_metric")
            ycol = "Count" if metric == "Raw mentions" else "Per 1,000 words"
            ont_chart = alt.Chart(ontology_counts).mark_bar().encode(
                x=alt.X("Category:N", title="Hotel aspect", axis=alt.Axis(labelAngle=0),
                        sort=list(ontology)),
                xOffset=alt.XOffset("Subclass:N", sort=["Negative", "Positive"]),
                y=alt.Y(f"{ycol}:Q", title=metric),
                color=alt.Color("Subclass:N", sort=["Negative", "Positive"]),
                tooltip=["Subclass:N", "Category:N", "Count:Q",
                         alt.Tooltip("Per 1,000 words:Q", format=".2f"), "Total words:Q"],
            ).properties(height=370)
            st.altair_chart(ont_chart, width="stretch")
            st.caption(
                "Negative and Positive bars are side by side for each category. "
                "The underlying counts are the number of word/phrase occurrences."
            )
            table_counts = (ontology_counts.pivot(index="Category", columns="Subclass", values="Count")
                            .reindex(list(ontology)).reset_index())
            st.write("**Exact term counts by subclass**")
            st.dataframe(table_counts, width="stretch", hide_index=True)
            st.caption(
                "Comparing normalized frequencies helps avoid interpreting a larger "
                "subclass corpus as a greater interest in an aspect. Word-list matching "
                "cannot resolve context, negation or the meaning of ambiguous words."
            )
    else:
        st.info("Task 11 needs review text and subclass labels to generate a current chart.")

with tabs[4]:
    st.subheader("Tasks 12–13 · LLM performance and review contradictions")
    st.write(
        "**Task 12** compares how accurately Qwen and SentiStrength classify "
        "hotel reviews as Positive or Negative. **Task 13** examines reviews "
        "where the sentiment of the text disagrees strongly with the customer's rating."
    )

    # TASK 12 – Only accuracy, plus separate prediction tables for both methods.
    st.markdown("### Task 12 · Qwen versus SentiStrength")
    st.write(
        "We compare each prediction with the customer's star rating. "
        "Ratings of **1–2 are Negative** and **4–5 are Positive**. "
        "Three-star reviews are not included in this test. "
        "Both methods analyze the review text without seeing the customer's rating."
    )
    llm = load_csv("task12_results.csv")
    if llm is not None and {"rating", "overall"}.issubset(llm.columns):
        experiment = llm.copy()
        experiment["rating"] = pd.to_numeric(experiment["rating"], errors="coerce")
        experiment["overall"] = pd.to_numeric(experiment["overall"], errors="coerce")
        experiment["Reference"] = experiment["rating"].map(rating_sentiment)
        experiment = experiment[
            experiment["Reference"].isin(["Negative", "Positive"])
        ].copy()

        if "sentistrength_label" in experiment.columns:
            experiment["SentiStrength"] = (
                experiment["sentistrength_label"].fillna("Invalid")
                .astype(str).str.strip().str.capitalize()
            )
        else:
            experiment["SentiStrength"] = experiment["overall"].map(sentiment_from_overall)

        llm_col = ("llm_label" if "llm_label" in experiment.columns else
                   "qwen_label" if "qwen_label" in experiment.columns else None)
        if llm_col is not None:
            experiment["Qwen"] = (
                experiment[llm_col].fillna("Invalid")
                .astype(str).str.strip().str.capitalize()
            )
            ss_accuracy = get_accuracy(experiment["Reference"], experiment["SentiStrength"])
            qwen_accuracy = get_accuracy(experiment["Reference"], experiment["Qwen"])
            if ss_accuracy is not None and qwen_accuracy is not None:
                reviews_col, senti_col, qwen_col = st.columns(3)
                reviews_col.metric("Test reviews", f"{len(experiment):,}")
                senti_col.metric("SentiStrength accuracy", f"{ss_accuracy:.1f}%")
                qwen_col.metric("Qwen accuracy", f"{qwen_accuracy:.1f}%")
                st.write(
                    "**Accuracy** shows the percentage of reviews that were classified "
                    "correctly compared with their customer rating."
                )

                st.markdown("#### Task 12 · Prediction outcomes")
                st.write(
                    "Each row represents the customer's actual rating class. "
                    "The columns show the model's predictions. "
                    "**Correct (%)** is the percentage of reviews in that row classified correctly."
                )
                for method in ["SentiStrength", "Qwen"]:
                    st.markdown(f"**{method} predictions**")
                    matrix = pd.crosstab(
                        experiment["Reference"], experiment[method]
                    ).reindex(index=["Negative", "Positive"], fill_value=0)
                    matrix.index.name = "Actual rating"
                    ordered = ["Negative", "Positive", "Neutral", "Invalid"]
                    extras = [col for col in matrix.columns if col not in ordered]
                    matrix = matrix.reindex(
                        columns=ordered + extras, fill_value=0
                    )
                    totals = matrix.sum(axis=1)
                    correct = pd.Series(
                        [matrix.loc[label, label] for label in ["Negative", "Positive"]],
                        index=["Negative", "Positive"],
                    )
                    matrix["Correct (%)"] = (
                        100 * correct.div(totals.replace(0, np.nan))
                    ).round(1)
                    st.dataframe(matrix.reset_index(), width="stretch", hide_index=True)

                st.markdown("#### Task 12 · Inspect review predictions")
                st.write(
                    "Choose which reviews to inspect. The table shows the customer "
                    "rating and the two predictions side by side."
                )
                filter_choice = st.selectbox(
                    "Display", ["All reviews", "Qwen errors", "SentiStrength errors",
                                "Models disagree", "SentiStrength unclassified"],
                    key="evaluation_filter",
                )
                selected = experiment.copy()
                if filter_choice == "Qwen errors":
                    selected = selected[selected["Qwen"] != selected["Reference"]]
                elif filter_choice == "SentiStrength errors":
                    selected = selected[selected["SentiStrength"] != selected["Reference"]]
                elif filter_choice == "Models disagree":
                    selected = selected[selected["Qwen"] != selected["SentiStrength"]]
                elif filter_choice == "SentiStrength unclassified":
                    selected = selected[
                        ~selected["SentiStrength"].isin(["Negative", "Positive"])
                    ]
                display_cols = [col for col in [
                    "review_id", "rating", "Reference", "overall", "SentiStrength", "Qwen", "text"
                ] if col in selected.columns]
                st.caption(f"Showing {len(selected):,} of {len(experiment):,} tested reviews.")
                st.dataframe(
                    selected[display_cols].reset_index(drop=True),
                    width="stretch", height=370, hide_index=True,
                )
            else:
                st.info("No valid reviews are available for the accuracy comparison.")
        else:
            st.warning("task12_results.csv needs an llm_label or qwen_label column.")
    elif llm is None:
        st.info("Place task12_results.csv in the project folder to show the saved results.")
    else:
        st.warning("task12_results.csv needs rating and overall columns.")

    st.divider()

    # TASK 13 – Only the active thresholds are shown to the user.
    st.markdown("### Task 13 · Rating–sentiment contradictions")
    st.write(
        "We flag **Case A** when the customer rating is **4–5** but the "
        "SentiStrength overall score is **−2 or lower**. We flag **Case B** "
        "when the rating is **1–2** but the overall score is **+2 or higher**."
    )

    source_data = df
    source_name = "D1.csv"
    if source_data is None or not {"rating", "overall"}.issubset(source_data.columns):
        source_data = load_csv("task13_bias_cases.csv")
        source_name = "task13_bias_cases.csv"
        if source_data is not None:
            st.warning(
                "D1.csv was not available, so the results were calculated from "
                "task13_bias_cases.csv. Include D1.csv to analyze the full dataset."
            )

    if source_data is not None and {"rating", "overall"}.issubset(source_data.columns):
        cases = prepare_bias_cases(source_data, min_strength=2)
        if cases.empty:
            st.info("No reviews meet these contradiction rules.")
        else:
            a_total = int(cases["Case type"].eq("Case A").sum())
            b_total = int(cases["Case type"].eq("Case B").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Contradictory reviews", f"{len(cases):,}")
            c2.metric("Case A · High rating", a_total)
            c3.metric("Case B · Low rating", b_total)
            if df is not None and len(df):
                st.caption(
                    f"{len(cases):,} of {len(df):,} reviews "
                    f"({100 * len(cases) / len(df):.2f}% of the dataset) meet these rules."
                )

            count_data = pd.DataFrame({
                "Case": ["Case A", "Case B"],
                "Number of cases": [a_total, b_total],
            })
            chart = alt.Chart(count_data).mark_bar().encode(
                x=alt.X("Case:N", title="Contradiction type",
                        sort=["Case A", "Case B"], axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Number of cases:Q", title="Number of reviews",
                        axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("Case:N", legend=None),
                tooltip=["Case:N", "Number of cases:Q"],
            ).properties(height=300)
            st.altair_chart(chart, width="stretch")
            st.caption(
                "Case A: a high customer rating but negative text sentiment. "
                "Case B: a low customer rating but positive text sentiment."
            )

            # Match Qwen outputs by review_id; missing predictions are not counted.
            cases, files_used = attach_bias_qwen(cases, [
                "task13_qwen_results_expanded.csv", "task13_qwen_results.csv",
                "task13_bias_cases.csv",
            ])
            valid = cases["Qwen"].isin(["Positive", "Negative"])
            # An explicit Invalid result is an evaluated but incorrect prediction.
            available = cases["Qwen"].isin(["Positive", "Negative", "Invalid"])
            agrees_rating = valid & cases["Qwen"].eq(cases["Rating class"])
            agrees_senti = valid & cases["Qwen"].eq(cases["SentiStrength"])
            evaluated_count = int(available.sum())

            st.markdown("#### Task 13 · Qwen comparison results")
            st.write(
                "Qwen reads the hotel review text and compares its prediction with "
                "the customer's rating and SentiStrength."
            )
            if evaluated_count == len(cases):
                comparison_results = pd.DataFrame({
                    "Comparison": ["Qwen matches user rating", "Qwen matches SentiStrength"],
                    "Number of reviews": [
                        f"{int(agrees_rating.sum())} / {len(cases)}",
                        f"{int(agrees_senti.sum())} / {len(cases)}",
                    ],
                    "Agreement (%)": [
                        round(100 * agrees_rating.sum() / len(cases), 1),
                        round(100 * agrees_senti.sum() / len(cases), 1),
                    ],
                })
            else:
                comparison_results = pd.DataFrame({
                    "Comparison": ["Qwen matches user rating", "Qwen matches SentiStrength"],
                    "Number of reviews": ["—", "—"],
                    "Agreement (%)": ["—", "—"],
                })
                st.info("Qwen predictions are incomplete for these contradiction cases.")
            st.dataframe(comparison_results, width="stretch", hide_index=True)

            by_case = cases.assign(
                **{"Qwen matches rating": agrees_rating}
            ).groupby("Case type", as_index=False).agg(
                **{
                    "Number of cases": ("rating", "size"),
                    "Qwen matches rating": ("Qwen matches rating", "sum"),
                }
            )
            if evaluated_count == len(cases):
                by_case["Qwen agrees with rating (%)"] = (
                    100 * by_case["Qwen matches rating"] / by_case["Number of cases"]
                ).round(1)
            else:
                by_case["Qwen matches rating"] = "—"
                by_case["Qwen agrees with rating (%)"] = "—"
            st.markdown("#### Task 13 · Qwen agreement by contradiction type")
            st.dataframe(by_case, width="stretch", hide_index=True)

            st.markdown("#### Task 13 · Review individual contradictions")
            cfilter, qfilter = st.columns(2)
            with cfilter:
                case_filter = st.selectbox(
                    "Filter by case", ["All", "Case A", "Case B"], key="bias_case_filter"
                )
            with qfilter:
                agreement_filter = st.selectbox(
                    "Filter by Qwen result",
                    ["All", "Qwen agrees with rating", "Qwen agrees with SentiStrength"],
                    key="bias_qwen_filter",
                )
            shown = cases.copy()
            if case_filter != "All":
                shown = shown[shown["Case type"].eq(case_filter)]
            if agreement_filter == "Qwen agrees with rating":
                shown = shown.loc[agrees_rating.reindex(shown.index, fill_value=False)]
            elif agreement_filter == "Qwen agrees with SentiStrength":
                shown = shown.loc[agrees_senti.reindex(shown.index, fill_value=False)]

            columns = [col for col in [
                "review_id", "hotel_id", "rating", "overall", "Case type",
                "Rating class", "SentiStrength", "Qwen", "text",
            ] if col in shown.columns]
            st.caption(f"Showing {len(shown):,} of {len(cases):,} contradictory reviews; source: {source_name}.")
            st.dataframe(
                shown[columns].reset_index(drop=True),
                width="stretch", height=450, hide_index=True,
            )
    else:
        st.info("Add D1.csv with rating and overall columns to analyze Task 13.")

