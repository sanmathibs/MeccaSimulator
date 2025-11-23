# tabs/tab_ai_insight.py
import os
import re
from collections import Counter
from glob import glob
from typing import Dict, List
from wordcloud import WordCloud
import matplotlib.pyplot as plt


import json

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import streamlit as st

from ai_engine import ask_ai, ensure_ai_session


# =========================
#  Coverage chips (forecast data)
# =========================
@st.cache_data(show_spinner=False)
def load_store_coverage():
    paths = glob("data/forecast/*.json")
    cards = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.json_normalize(data["Data"])
        df["Store"] = (
            os.path.basename(path).replace(".json", "").replace("_", " ")
        )
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        cards.append(
            {
                "store": df["Store"].iloc[0],
                "days": int(df["Date"].nunique()),
                "date_min": df["Date"].min().date(),
                "date_max": df["Date"].max().date(),
                "sales": float(df["SalesActual"].fillna(0).sum()),
            }
        )
    return cards


def render_data_pills():
    cards = load_store_coverage()
    html = "<div class='ai-pill-bar'>"
    for c in cards:
        html += (
            "<div class='ai-pill'>"
            f"<strong>{c['store']}</strong> · {c['days']} days · "
            f"{c['date_min']} – {c['date_max']}"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =========================
#  Review / competitor summary (for pills)
# =========================
@st.cache_data(show_spinner=False)
def load_review_summary() -> pd.DataFrame:
    """
    Aggregate Google-style review JSON (list of reviews) into one row per store.

    Expected per-file structure (simplified):
    [
      {
        "title": "MECCA Double Bay",
        "stars": 5,
        "reviewsCount": 59,
        "totalScore": 4.5,
        ...
      },
      ...
    ]
    """
    rows: List[Dict] = []

    for path in glob("data/Reviews/*.json"):
        with open(path, "r", encoding="utf-8") as f:
            reviews = json.load(f)

        if not reviews:
            continue

        first = reviews[0]
        store = first.get("title") or os.path.splitext(os.path.basename(path))[0]

        star_values = [
            r.get("stars")
            for r in reviews
            if isinstance(r.get("stars"), (int, float))
        ]

        total_scores = [
            r.get("totalScore")
            for r in reviews
            if isinstance(r.get("totalScore"), (int, float))
        ]
        if total_scores:
            rating = float(total_scores[0])
        elif star_values:
            rating = float(sum(star_values) / len(star_values))
        else:
            rating = None

        counts = [
            r.get("reviewsCount")
            for r in reviews
            if isinstance(r.get("reviewsCount"), (int, float))
        ]
        review_count = int(max(counts)) if counts else len(reviews)

        if star_values:
            pos_share = sum(1 for s in star_values if s >= 4) / len(star_values)
            neg_share = sum(1 for s in star_values if s <= 2) / len(star_values)
        else:
            pos_share = None
            neg_share = None

        brand = "Mecca" if "MECCA" in store.upper() else "Competitor"
        location = store.split(" ", 1)[1] if " " in store else store

        rows.append(
            {
                "store": store,
                "brand": brand,
                "location": location,
                "rating": rating,
                "review_count": review_count,
                "pos_share": pos_share,
                "neg_share": neg_share,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def render_review_pills():
    df = load_review_summary()
    if df.empty:
        return

    # 🔒 This tab is Mecca-only
    df = df[df["brand"] == "Mecca"].copy()
    if df.empty:
        return

    st.markdown("#### 🗣️ Customer pulse (Mecca stores)")

    # Order by location for a nice strip
    df = df.sort_values(["location"], ascending=[True])

    html = "<div class='ai-review-bar'>"
    for _, row in df.iterrows():
        rating = row["rating"]
        count = row["review_count"]
        store = row["store"]

        html += f"""
        <div class="ai-review-pill">
            <div class="ai-review-header">
                <span class="ai-review-badge ai-review-badge-mecca">
                    MECCA
                </span>
                <span class="ai-review-store">{store}</span>
            </div>
            <div class="ai-review-main">
                ⭐ {rating:.1f} · {count} reviews
            </div>
        """

        pos = row.get("pos_share")
        neg = row.get("neg_share")
        if pos is not None and neg is not None:
            html += (
                f"<div class='ai-review-extra'>"
                f"{pos*100:,.0f}% positive · {neg*100:,.0f}% negative"
                f"</div>"
            )

        html += "</div>"

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)



# =========================
#  Shared helpers
# =========================
def _render_view_suggestion(view_info: Dict):
    st.markdown(
        f"""
        <div class="ai-view-suggestion">
            <strong>Recommended view:</strong> {view_info['tab_label']}<br/>
            <span class="ai-view-reason">{view_info['reason']}</span><br/>
            <span class="ai-view-note">
                Use the navigation tabs at the top to jump there for charts & KPIs.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_customer_evidence(snippets: List[Dict]):
    review_snips = [
        s for s in snippets if "review" in (s.get("type", "") or "").lower()
    ]
    if not review_snips:
        return

    with st.expander("🧾 Sample customer quotes used", expanded=False):
        for s in review_snips[:6]:
            st.markdown(f"- {s['text']}")



# =========================
#  Review + ops helper data
# =========================
@st.cache_data(show_spinner=False)
def load_reviews_long() -> pd.DataFrame:
    """
    Load all Google-review style JSON files from data/Reviews (and data/reviews)
    into one long dataframe.

    Expected fields (as in your example):
    reviewId, text, stars, name, publishedAtDate, title, totalScore, ...
    """
    paths = glob("data/Reviews/*.json") + glob("data/reviews/*.json")
    frames = []
    if not paths:
        return pd.DataFrame()

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        store_file = os.path.basename(path).replace(".json", "")
        df["store_label"] = store_file

        upper = store_file.upper()
        if upper.startswith("MECCA"):
            brand = "MECCA"
        elif upper.startswith("SEPHORA"):
            brand = "SEPHORA"
        else:
            brand = "Competitor"

        df["brand"] = brand

        loc = store_file
        for prefix in ["MECCA", "SEPHORA", "W Cosmetics"]:
            loc = loc.replace(prefix, "")
        df["location"] = loc.strip()

        df["publishedAtDate"] = pd.to_datetime(
            df.get("publishedAtDate"), errors="coerce"
        )
        df["stars"] = pd.to_numeric(df.get("stars"), errors="coerce")

        def star_to_sentiment(x):
            if pd.isna(x):
                return "unknown"
            if x >= 4:
                return "positive"
            if x <= 2:
                return "negative"
            return "neutral"

        df["sentiment"] = df["stars"].apply(star_to_sentiment)

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_store_ops_kpis() -> pd.DataFrame:
    """
    Very light per-store ops KPIs used to relate reviews to performance:
    - MECCA forecast MAPE
    - Sales per labour hour
    """
    paths = glob("data/forecast/*.json")
    frames = []
    if not paths:
        return pd.DataFrame()

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.json_normalize(data["Data"])
        df["Store"] = os.path.basename(path).replace(".json", "").replace("_", " ")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    rows = []
    for store, g in df.groupby("Store"):
        hist = g[g["SalesActual"].notna()].copy()
        if hist.empty:
            continue

        hist["abs_err_mecca"] = (hist["SalesActual"] - hist["SalesForecast"]).abs()
        mape = (
            hist["abs_err_mecca"]
            .div(hist["SalesActual"].replace(0, np.nan))
            .mean()
            * 100
        )

        total_sales = hist["SalesActual"].sum()
        total_hours = hist["HoursActual"].fillna(0).sum()
        sph = total_sales / total_hours if total_hours else np.nan

        rows.append(
            {
                "store_label": store,
                "mape_mecca": float(mape),
                "sales_per_hour": float(sph),
            }
        )

    return pd.DataFrame(rows)


STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "mecca",
    "store",
    "shop",
    "one",
    "also",
    "really",
    "great",
    "good",
    "nice",
    "cosmetics",
    "makeup",
}


def build_keyword_df(
    df_reviews: pd.DataFrame,
    sentiment_filter: str | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Cheap 'word map' builder: top tokens for given sentiment bucket.
    """
    ser = df_reviews["text"].dropna().astype(str)

    if sentiment_filter:
        ser = df_reviews.loc[
            df_reviews["sentiment"] == sentiment_filter, "text"
        ].dropna().astype(str)

    counter = Counter()
    for txt in ser:
        tokens = re.findall(r"[A-Za-z']+", txt.lower())
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
        counter.update(tokens)

    if not counter:
        return pd.DataFrame(columns=["token", "count"])

    common = counter.most_common(top_n)
    return pd.DataFrame(common, columns=["token", "count"])


def cx_kpi_card(
    title: str,
    value: str,
    subtitle: str | None = None,
    variant: str | None = None,
) -> str:
    """
    variant:
      - 'rating-high'  -> rating >= 4.5  (green, #31572c)
      - 'rating-mid'   -> rating < 4.5   (green, #4f772d)
      - 'positive'     -> good KPI (e.g. high positive share)
      - 'negative'     -> bad KPI (e.g. high negative share)
      - None           -> neutral styling
    """
    extra_class = f" cx-kpi-{variant}" if variant else ""
    subtitle_html = f"<div class='cx-kpi-sub'>{subtitle}</div>" if subtitle else ""
    return f"""
    <div class="cx-kpi-card{extra_class}">
        <div class="cx-kpi-label">{title}</div>
        <div class="cx-kpi-value">{value}</div>
        {subtitle_html}
    </div>
    """



def cx_store_card(
    store: str,
    rating: float,
    reviews: int,
    pos_share: float,
    neg_share: float,
) -> str:
    return f"""
    <div class="cx-store-card">
        <div class="cx-store-name">{store}</div>
        <div class="cx-store-rating">⭐ {rating:.2f}</div>
        <div class="cx-store-metrics">
            <span>{reviews:,} reviews</span>
            <span>{pos_share:.1f}% positive</span>
            <span>{neg_share:.1f}% negative</span>
        </div>
    </div>
    """


# -------------------------
#  Helpers for CX view
# -------------------------
def pick_sample_reviews(
    df_store: pd.DataFrame,
    sentiment: str,
    max_reviews: int = 3,
    min_words: int = 8,
) -> list[str]:
    """
    Pick up to `max_reviews` unique review texts for a store & sentiment,
    keeping only reviews that look like full sentences (min_words).
    """
    ser = (
        df_store.loc[df_store["sentiment"] == sentiment, "text"]
        .dropna()
        .astype(str)
    )

    # Only keep "real" sentences
    ser = ser[ser.str.split().str.len() >= min_words]

    # Drop exact duplicates to avoid repetition
    ser = ser.drop_duplicates()

    return ser.head(max_reviews).tolist()


def render_wordcloud_for_subset(
    df_subset: pd.DataFrame,
    sentiment_filter: str,
    key: str,
    max_words: int = 50,
):
    """Render a word cloud image for the given subset + sentiment."""
    df_subset = df_subset.copy()
    if sentiment_filter:
        df_subset = df_subset[df_subset["sentiment"] == sentiment_filter]

    texts = df_subset["text"].dropna().astype(str).tolist()
    texts = [t for t in texts if len(t.strip()) > 20]  # at least a short sentence

    if not texts:
        st.caption("Not enough text to build a word cloud.")
        return

    full_text = " ".join(texts)
    wc = WordCloud(
        width=400,
        height=300,
        background_color= "White",
        stopwords=STOPWORDS,
        max_words=max_words,
        collocations=False,
    ).generate(full_text)

    st.image(wc.to_image(), width='content')


def select_sample_reviews(df_store: pd.DataFrame, sentiment: str, n: int) -> pd.DataFrame:
    """Pick up to n reasonably long reviews for a given sentiment."""
    df = df_store[df_store["sentiment"] == sentiment].copy()
    df["text"] = df["text"].astype(str)
    # keep only reviews that are at least a sentence-ish
    df = df[df["text"].str.len() >= 60]
    # prioritise more recent & more extreme ratings
    df = df.sort_values(
        by=["publishedAtDate", "stars"],
        ascending=[False, sentiment != "negative"],
    )
    return df.head(n)


def highlight_keywords(text: str, keywords: set[str]) -> str:
    """Highlight important tokens inside a review with <mark>."""
    if not keywords:
        return text

    # only simple words, avoid regex surprises
    safe_keywords = [re.escape(k) for k in keywords if len(k) > 2]
    if not safe_keywords:
        return text

    pattern = r"\\b(" + "|".join(safe_keywords) + r")\\b"

    def _repl(match):
        return f"<mark>{match.group(0)}</mark>"

    return re.sub(pattern, _repl, text, flags=re.IGNORECASE)

def render_review_intelligence():
    """
    Deep-dive Mecca customer reviews: filters + word clouds + sentiment dashboard + AI Q&A.
    Mecca only, up to 3 stores compared.
    """
    df_all = load_reviews_long()
    if df_all.empty:
        st.info("No review data found in data/Reviews or data/reviews.")
        return

    # --- Mecca only + basic cleaning / de-dup ---------------------------------
    df_all = df_all[df_all["brand"] == "MECCA"].copy()
    if df_all.empty:
        st.info("No Mecca review data found.")
        return

    df_all["text"] = df_all["text"].fillna("").astype(str)
    df_all["n_words"] = df_all["text"].str.split().str.len()
    df_all = df_all[df_all["n_words"] >= 8]  # only real sentences
    df_all = df_all.drop_duplicates(subset=["store_label", "text"])

    # helper: nicer wordcloud with fewer words
    def draw_wordcloud(series: pd.Series, title: str, key: str):
        """Compact, softer word-cloud so it doesn't dwarf other charts."""
        text_blob = " ".join(series.dropna().astype(str).tolist())
        if not text_blob.strip():
            st.caption(f"Not enough text to build a word cloud for {title}.")
            return

        wc = WordCloud(
            width=480,
            height=220,                # much shorter canvas
            background_color="#f8fafc",# soft blue-grey background
            stopwords=STOPWORDS,
            max_words=60,              # fewer words => less busy
            collocations=False,
            prefer_horizontal=0.95,
            relative_scaling=0.4,
            max_font_size=42,          # cap biggest words
        ).generate(text_blob)

        fig, ax = plt.subplots(figsize=(3.6, 1.8), dpi=100)  # small physical size
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=12, pad=6)
        st.pyplot(fig, clear_figure=True, width='content')

    # helper: highlight keywords inside reviews
    def highlight_keywords(text: str, keywords: set[str]) -> str:
        if not keywords:
            return text
        escaped = [re.escape(k) for k in sorted(keywords, key=len, reverse=True)]
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        return re.sub(
            pattern,
            r"<span class='cx-keyword'>\1</span>",
            text,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------------ layout
    st.markdown("### 🗣️ Customer review intelligence")
    st.caption(
        "Explore Google reviews for Mecca stores, compare locations, and let Mecca AI "
        "surface the key customer themes."
    )

    # You can still show the review pills strip if you like
    render_review_pills()

    # ----------------------------- filters ------------------------------------
    col_stores, col_window = st.columns([2.5, 1.1])

    all_labels = sorted(df_all["store_label"].unique())
    default_focus = all_labels[:3]

    stores_in_focus = col_stores.multiselect(
        "Mecca stores in focus (max 3 used for comparison)",
        options=all_labels,
        default=default_focus,
        key="cx_mecca_store_focus",
    )
    if not stores_in_focus:
        stores_in_focus = all_labels[:1]
    if len(stores_in_focus) > 3:
        stores_in_focus = stores_in_focus[:3]
        st.warning("Only the first 3 stores are used for comparison.")

    window = col_window.selectbox(
        "Review window",
        options=["All time", "Last 3 months", "Last 6 months", "Last 12 months"],
        index=0,
        key="cx_mecca_window",
    )

    df = df_all[df_all["store_label"].isin(stores_in_focus)].copy()
    if df.empty:
        st.info("No reviews match the selected filters.")
        return

    if window != "All time":
        months = int(window.split()[1])
        max_date = df["publishedAtDate"].max()
        if pd.notna(max_date):
            cutoff = max_date - pd.DateOffset(months=months)
            df = df[df["publishedAtDate"] >= cutoff]

    if df.empty:
        st.info("No reviews in the chosen time window.")
        return

    # ----------------------------- mode: single vs compare --------------------
    col_mode, col_store_sel = st.columns([1.0, 2.5])
    mode = col_mode.radio(
        "Mode",
        options=["Single store", "Compare stores"],
        horizontal=True,
        key="cx_mecca_mode",
    )

    if mode == "Single store":
        store_list = stores_in_focus
        store_a = col_store_sel.selectbox(
            "Store A",
            options=store_list,
            index=0,
            key="cx_mecca_store_a",
        )
        compare_stores = [store_a]
    else:
        compare_stores = col_store_sel.multiselect(
            "Stores to compare (up to 3)",
            options=stores_in_focus,
            default=stores_in_focus,
            key="cx_mecca_store_compare",
        )
        if not compare_stores:
            compare_stores = stores_in_focus[:1]
        if len(compare_stores) > 3:
            compare_stores = compare_stores[:3]
            st.warning("Comparison limited to the first 3 selected stores.")

    df_cmp = df[df["store_label"].isin(compare_stores)].copy()

    # ----------------------------- KPI strip ----------------------------------
    avg_rating = df_cmp["stars"].mean()
    n_reviews = len(df_cmp)
    pos_share = (df_cmp["sentiment"] == "positive").mean() * 100
    neg_share = (df_cmp["sentiment"] == "negative").mean() * 100

    # choose colour variants for KPIs
    rating_variant = "rating-high" if avg_rating >= 4.5 else "rating-mid"
    pos_variant = "positive" if pos_share >= 80 else None
    neg_variant = "negative" if neg_share >= 15 else None

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            cx_kpi_card("Avg rating", f"{avg_rating:.2f} ⭐", variant=rating_variant),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            cx_kpi_card("Reviews in view", f"{n_reviews:,}", "All sentiments"),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            cx_kpi_card(
                "Positive share",
                f"{pos_share:.1f}%",
                "4–5★ reviews",
                variant=pos_variant,
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            cx_kpi_card(
                "Negative share",
                f"{neg_share:.1f}%",
                "1–2★ reviews",
                variant=neg_variant,
            ),
            unsafe_allow_html=True,
        )

    # ----------------------------- Store snapshot -----------------------------
    st.markdown("#### 🧾 Store comparison snapshot")

    snapshot = (
        df_cmp.groupby("store_label")
        .agg(
            avg_rating=("stars", "mean"),
            review_count=("reviewId", "count"),
            pos_share=("sentiment", lambda s: (s == "positive").mean() * 100),
            neg_share=("sentiment", lambda s: (s == "negative").mean() * 100),
        )
        .reset_index()
    )

    cols = st.columns(len(compare_stores))
    for store, col in zip(compare_stores, cols):
        row = snapshot[snapshot["store_label"] == store].iloc[0]
        subtitle = (
            f"{int(row['review_count']):,} reviews  ·  "
            f"{row['pos_share']:.1f}% positive  ·  {row['neg_share']:.1f}% negative"
        )
        variant = "rating-high" if row["avg_rating"] >= 4.5 else "rating-mid"
        with col:
            st.markdown(
                cx_kpi_card(
                    row["store_label"],
                    f"{row['avg_rating']:.2f} ⭐",
                    subtitle,
                    variant=variant,
                ),
                unsafe_allow_html=True,
            )

    # ----------------------------- Sentiment mix chart ------------------------
    st.markdown("#### 📊 Sentiment mix by store (share of reviews)")

    sent = (
        df_cmp.groupby(["store_label", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    totals = sent.groupby("store_label")["count"].transform("sum")
    sent["pct"] = sent["count"] / totals * 100
    sent["sentiment"] = pd.Categorical(
        sent["sentiment"], categories=["positive", "neutral", "negative"], ordered=True
    )

    fig_sent = px.bar(
        sent,
        x="store_label",
        y="pct",
        color="sentiment",
        text="pct",
        labels={"store_label": "Store", "pct": "% of reviews", "sentiment": ""},
        barmode="stack",
        color_discrete_map={
            "positive": "#679436",  # green
            "neutral": "#d4d4d8",
            "negative": "#a63c06",  # warm negative
        },
    )
    fig_sent.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    fig_sent.update_layout(
        yaxis=dict(range=[0, 100]),
        margin=dict(l=10, r=10, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_sent, width='content')

    # ----------------------------- Rating distribution ------------------------
    st.markdown("#### ⭐ Rating distribution by store")

    df_r = df_cmp[df_cmp["stars"].notna()].copy()
    if not df_r.empty:
        df_r["star_bucket"] = (
            df_r["stars"].round().clip(lower=1, upper=5).astype(int)
        )

        dist = (
            df_r.groupby(["store_label", "star_bucket"])
            .size()
            .reset_index(name="count")
        )
        dist["pct"] = dist["count"] / dist.groupby("store_label")["count"].transform(
            "sum"
        ) * 100
        dist["star_bucket"] = dist["star_bucket"].astype(str)

        fig_dist = px.bar(
            dist,
            x="star_bucket",
            y="pct",
            color="store_label",
            text="pct",
            labels={
                "star_bucket": "Star rating",
                "pct": "% of reviews",
                "store_label": "Store",
            },
            barmode="group",
            color_discrete_sequence=["#003459", "#679436", "#4f772d"],
        )
        fig_dist.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_dist.update_layout(
            yaxis=dict(range=[0, 100]),
            margin=dict(l=10, r=10, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dist, width='stretch')
    else:
        st.info("No star ratings available to build a distribution.")

    # ----------------------------- Word clouds + top words --------------------
    st.markdown("#### ☁️ Word clouds by store")

    wc_cols = st.columns(len(compare_stores))
    for store, col in zip(compare_stores, wc_cols):
        with col:
            df_store = df_cmp[df_cmp["store_label"] == store].copy()

            # Top repeated words (all sentiments)
            kw = build_keyword_df(df_store, top_n=6)
            if not kw.empty:
                chips_html = "<div class='cx-top-words'>"
                for _, r in kw.iterrows():
                    token = r["token"]
                    cnt = int(r["count"])
                    chips_html += (
                        f"<span class='cx-top-word-chip'>{token}"
                        f"<span class='cx-top-word-count'>{cnt}</span></span>"
                    )
                chips_html += "</div>"
                st.markdown(chips_html, unsafe_allow_html=True)

            draw_wordcloud(df_store["text"], title=store, key=f"wc_{store}")

    # ----------------------------- Sample reviews -----------------------------
    st.markdown("#### ✏️ Sample reviews by store")

    max_reviews = st.slider(
        "Number of reviews per bucket",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        key="cx_mecca_n_reviews",
    )

    for store in compare_stores:
        df_store = df_cmp[df_cmp["store_label"] == store].copy()
        df_store = df_store[df_store["n_words"] >= 8].drop_duplicates(subset=["text"])

        kw_tokens = set(
            build_keyword_df(df_store, top_n=20)["token"].str.lower().tolist()
        )

        top_pos = (
            df_store[df_store["sentiment"] == "positive"]
            .sort_values(["stars", "n_words"], ascending=[False, False])
            .head(max_reviews)
        )
        top_neg = (
            df_store[df_store["sentiment"] == "negative"]
            .sort_values(["stars", "n_words"], ascending=[True, False])
            .head(max_reviews)
        )

        with st.expander(f"{store}: sample reviews", expanded=False):
            st.markdown("**⭐ Top positive**")
            if top_pos.empty:
                st.write("_No clearly positive reviews in view._")
            else:
                for _, row in top_pos.iterrows():
                    txt = highlight_keywords(row["text"], kw_tokens)
                    st.markdown(
                        f"- ⭐ **{int(row['stars'])}** – {txt}",
                        unsafe_allow_html=True,
                    )

            st.markdown("**⚠️ Top negative**")
            if top_neg.empty:
                st.write("_No clearly negative reviews in view._")
            else:
                for _, row in top_neg.iterrows():
                    txt = highlight_keywords(row["text"], kw_tokens)
                    st.markdown(
                        f"- ⭐ **{int(row['stars'])}** – {txt}",
                        unsafe_allow_html=True,
                    )

    # ----------------------------- Free-form AI Q&A ---------------------------
    st.markdown("### 🤖 Ask Mecca AI for more review insights")

    q_free = st.text_input(
        "Ask about anything you see in the review dashboards "
        "(staff, queues, events, returns, specific stores, etc.)",
        key="cx_mecca_free_q",
        placeholder="E.g. What do reviews suggest about staffing levels at Parramatta vs Castle Towers?",
    )

    if st.button("🚀 Ask Mecca AI (reviews + ops)", key="cx_mecca_free_btn") and q_free.strip():
        with st.spinner("Thinking..."):
            answer, snippets, _ = ask_ai(
                q_free
                + "\nUse both customer reviews and operational KPIs where helpful. "
                "Do not invent specific numbers that aren't implied by the context.",
                response_mode="detailed",
                deep_dive=True,
                retrieval_mode="blended_ops_cx",
            )
        st.markdown("##### Mecca AI answer")
        st.markdown(answer)
        if snippets:
            with st.expander("Context snippets used", expanded=False):
                for s in snippets:
                    st.markdown(f"- *{s.get('type','snippet')}*: {s['text']}")

    # keyword highlight CSS
    st.markdown(
        """
        <style>
        .cx-keyword {
            background: #fef3c7;
            padding: 0 2px;
            border-radius: 3px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )






# =========================
#  Quick Insights tab
# =========================
def render_quick_insights():
    st.subheader("⚡ One-click insights")

    st.caption(
        "Use these cards to instantly show why forecasting, rostering and customer "
        "experience matter for Mecca."
    )

    render_review_pills()

    resp_mode = st.radio(
        "Response length",
        options=["Short", "Detailed"],
        index=0,
        horizontal=True,
        key="quick_resp_mode",
    )
    deep = st.checkbox(
        "Deeper analysis (drivers & implications)",
        value=False,
        key="quick_deep",
    )

    focus_choice = st.radio(
        "Insight focus",
        options=[
            "Blended: ops + customer voice",
            "Operational metrics only (forecasts, rosters)",
            "Customer voice & market only",
        ],
        index=0,
        horizontal=True,
        key="quick_focus",
    )
    if focus_choice.startswith("Blended"):
        default_mode = "blended"
    elif focus_choice.startswith("Operational"):
        default_mode = "ops_only"
    else:
        default_mode = "customer_only"

    cards = [
        {
            "key": "qi_forecasting_value",
            "title": "Why forecasting matters",
            "subtitle": "Link forecast accuracy, rostering and what customers actually experience.",
            "prompt": (
                "Explain, using Mecca's context, why high-quality sales forecasting matters. "
                "Connect MAPE/MAE to staffing levels, wage control and customer queues, "
                "and use review snippets where relevant."
            ),
            "mode": "blended",
        },
        {
            "key": "qi_hybrid_vs_mecca",
            "title": "Hybrid vs MECCA uplift",
            "subtitle": "Show the value of the Hybrid model over MECCA baseline.",
            "prompt": (
                "Compare the Hybrid forecast to the MECCA baseline across stores. "
                "Quantify typical error reduction and translate this into fewer mis-staffed days, "
                "wage savings and better on-shelf availability."
            ),
            "mode": "ops_only",
        },
        {
            "key": "qi_store_health",
            "title": "Store health check",
            "subtitle": "Blend KPIs with review sentiment by store.",
            "prompt": (
                "Create a store health scorecard that combines forecast accuracy, "
                "sales per labour hour and customer review tone. Use letter grades (A–D) "
                "for each store and call out 3 key actions."
            ),
            "mode": "blended",
        },
        {
            "key": "qi_customer_love_pain",
            "title": "Customer love vs pain points",
            "subtitle": "What Mecca reviewers praise or complain about most.",
            "prompt": (
                "From Mecca store reviews, build two 'word maps': one for what "
                "customers love, one for pain points. Group themes such as staff, product "
                "range, price, wait times, events, etc. Highlight 3 opportunities for Mecca."
            ),
            "mode": "customer_only",
        },
        {
            "key": "qi_store_league",
            "title": "Store league table",
            "subtitle": "Rank Mecca stores by rating & sentiment.",
            "prompt": (
                "Rank Mecca stores by average Google rating, positive/negative review share, "
                "and review volume. Highlight the top 3 and bottom 3 stores, describe the key "
                "themes that make leaders stand out, and suggest 3 actions underperforming "
                "stores could take to catch up."
            ),
            "mode": "customer_only",
        },
        {
            "key": "qi_event_readiness",
            "title": "Event readiness",
            "subtitle": "Black Friday & Christmas vs business-as-usual.",
            "prompt": (
                "Compare event days (Black Friday, Christmas and similar) with non-event days. "
                "Comment on uplift, forecast accuracy on peaks, and how Mecca should plan rosters "
                "so reviews stay positive on the busiest days."
            ),
            "mode": "blended",
        },
        {
            "key": "qi_weekend_story",
            "title": "Weekend vs weekday story",
            "subtitle": "Weekend uplift, queues and staffing implications.",
            "prompt": (
                "Analyse weekend vs weekday sales and labour hours across stores. "
                "Explain what this means for roster design and how this lines up with "
                "weekend-specific customer feedback."
            ),
            "mode": "blended",
        },
        {
            "key": "qi_roster_hotspots",
            "title": "Roster risk hotspots",
            "subtitle": "Where over- or under-staffing hurts experience.",
            "prompt": (
                "Using roster and forecast context, highlight roles or weeks that look over- or "
                "under-staffed. Where you see negative reviews talking about queues or poor service, "
                "connect those to specific hotspots."
            ),
            "mode": "blended",
        },
        {
            "key": "qi_reviews_deep_dive",
            "title": "Customer reviews deep-dive",
            "subtitle": "Store-by-store improvement plan from 400+ reviews.",
            "prompt": (
                "Use Mecca review snippets plus operational context to produce a "
                "deep-dive on customer sentiment. For each Mecca store, summarise what is working, "
                "what is missing, and give 3 roster/forecasting ideas that would most improve reviews."
            ),
            "mode": "customer_only",
        },
    ]

    clicked_prompt = None
    clicked_mode = default_mode

    for row_start in range(0, len(cards), 3):
        cols = st.columns(3)
        for card, col in zip(cards[row_start : row_start + 3], cols):
            with col:
                clicked = st.button(card["title"], key=card["key"])
                st.markdown(
                    f"<div class='ai-q-sub'>{card['subtitle']}</div>",
                    unsafe_allow_html=True,
                )
            if clicked:
                clicked_prompt = card["prompt"]
                clicked_mode = card.get("mode", default_mode)
                st.session_state["quick_last_card"] = card["key"]

    if clicked_prompt:
        with st.spinner("Mecca AI is thinking..."):
            answer, snippets, view_info = ask_ai(
                clicked_prompt,
                response_mode="short" if resp_mode == "Short" else "detailed",
                deep_dive=deep,
                retrieval_mode=clicked_mode,
            )

        st.markdown("### 🧠 Insight")
        st.markdown(answer)
        _render_view_suggestion(view_info)

        with st.expander("🔍 Context snippets used", expanded=False):
            for s in snippets:
                st.markdown(f"- *{s.get('type','snippet')}*: {s['text']}")

        if clicked_mode in ("customer_only", "blended"):
            _render_customer_evidence(snippets)


# =========================
#  Conversational AI tab
# =========================
def render_conversational_ai():
    st.subheader("💬 Chat with Mecca AI")

    col_opts, col_hist = st.columns([1.3, 1])

    with col_opts:
        resp_mode = st.radio(
            "Response length",
            options=["Short", "Detailed"],
            index=0,
            horizontal=True,
            key="chat_resp_mode",
        )
        deep = st.checkbox(
            "Deeper analysis (drivers, outliers, methodology)",
            value=False,
            key="chat_deep",
        )
        focus_choice = st.radio(
            "Insight focus for this conversation",
            options=[
                "Blended: ops + customer voice",
                "Operational data only (forecasts, rosters)",
                "Customer voice & market only",
            ],
            index=0,
            horizontal=False,
            key="chat_focus",
        )
        if focus_choice.startswith("Blended"):
            chat_mode = "blended"
        elif focus_choice.startswith("Operational"):
            chat_mode = "ops_only"
        else:
            chat_mode = "customer_only"

    with col_hist:
        history = st.session_state.get("chat_history", [])
        if history:
            with st.expander("Conversation so far", expanded=False):
                for msg in history[-12:]:
                    if msg["role"] == "user":
                        st.markdown(f"**You:** {msg['content']}")
                    elif msg["role"] == "assistant":
                        st.markdown(f"**Mecca AI:** {msg['content']}")

    user_input = st.text_area(
        "Ask a question or follow-up:",
        placeholder=(
            "E.g., Compare Double Bay vs Parramatta for weekends, reviews and "
            "suggest roster changes."
        ),
        height=90,
    )

    col_send, col_clear = st.columns([0.7, 0.3])
    send_clicked = col_send.button("🚀 Ask Mecca AI", key="chat_send")
    clear_clicked = col_clear.button("🔄 Reset chat", key="chat_reset")

    if send_clicked and user_input.strip():
        with st.spinner("Thinking..."):
            answer, snippets, view_info = ask_ai(
                user_input,
                response_mode="short" if resp_mode == "Short" else "detailed",
                deep_dive=deep,
                retrieval_mode=chat_mode,
            )

        st.session_state["chat_last_answer"] = answer
        st.session_state["chat_last_snippets"] = snippets
        st.session_state["chat_last_view_info"] = view_info

    if clear_clicked:
        st.session_state["chat_history"] = []
        st.session_state["ai_session_started_at"] = None
        st.session_state["chat_last_answer"] = ""
        st.session_state["chat_last_snippets"] = []
        st.session_state["chat_last_view_info"] = None
        st.success("AI session cleared for this browser window.")

    last_answer = st.session_state.get("chat_last_answer", "")
    if last_answer:
        st.markdown("### 🧠 Mecca AI")
        st.markdown(last_answer)

        view_info = st.session_state.get("chat_last_view_info")
        if view_info:
            _render_view_suggestion(view_info)

        snippets = st.session_state.get("chat_last_snippets") or []
        if snippets:
            with st.expander("🔍 Context snippets used", expanded=False):
                for s in snippets:
                    st.markdown(f"- *{s.get('type','snippet')}*: {s['text']}")
            _render_customer_evidence(snippets)

    st.caption(
        "Chat history stays only in this browser window for about 1 hour and is then reset automatically."
    )


# =========================
#  Entry point
# =========================
def render():
    ensure_ai_session()

    st.markdown(
        """
        <style>
        .ai-pill-bar {
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin-bottom:6px;
        }
        .ai-pill {
            padding:4px 10px;
            border-radius:999px;
            background:linear-gradient(90deg,#eef2ff,#f9fafb);
            border:1px solid #e5e7eb;
            font-size:11px;
            color:#111827;
            white-space:nowrap;
        }
        
                /* KPI variants based on performance */
        .cx-kpi-card.cx-kpi-rating-high {
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
            border-color: #67943633;
        }
        .cx-kpi-card.cx-kpi-rating-mid {
            background: linear-gradient(135deg, #fefce8, #fef9c3);
            border-color: #4f772d33;
        }
        .cx-kpi-card.cx-kpi-positive {
            background: linear-gradient(135deg, #ecfdf5, #e0f2fe);
            border-color: #67943633;
        }
        .cx-kpi-card.cx-kpi-negative {
            background: linear-gradient(135deg, #fef2e2, #fee2e2);
            border-color: #a63c0633;
        }
        
        /* Rating colour accents */
        .cx-kpi-card.cx-kpi-rating-high .cx-kpi-value {
            color: #31572c;
        }
        .cx-kpi-card.cx-kpi-rating-mid .cx-kpi-value {
            color: #4f772d;
        }
        
        /* Top repeated words chips (above word clouds) */
        .cx-top-words {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 6px;
        }
        .cx-top-word-chip {
            padding: 3px 8px;
            border-radius: 999px;
            border: 1px solid #e5e7eb;
            background: #f9fafb;
            font-size: 11px;
            color: #003459;
        }
        .cx-top-word-count {
            margin-left: 4px;
            font-weight: 600;
            color: #003459;
        }


        .cx-kpi-card {
            padding: 12px 14px;
            border-radius: 10px;
            background: linear-gradient(135deg, #f9fafb, #eef2ff);
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
            margin-bottom: 10px;
        }
        .cx-kpi-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #6b7280;
            margin-bottom: 4px;
        }
        .cx-kpi-value {
            font-size: 20px;
            font-weight: 700;
            color: #111827;
        }
        .cx-kpi-sub {
            font-size: 11px;
            color: #9ca3af;
            margin-top: 2px;
        }

        .cx-store-card {
            padding: 10px 12px;
            border-radius: 12px;
            background: linear-gradient(135deg, #ecfdf5, #e0f2fe);
            border: 1px solid #bae6fd;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.08);
            margin-bottom: 10px;
            font-size: 11px;
        }
        .cx-store-name {
            font-weight: 600;
            font-size: 12px;
            color: #0f172a;
            margin-bottom: 2px;
        }
        .cx-store-rating {
            font-size: 16px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 2px;
        }
        .cx-store-metrics {
            display:flex;
            flex-wrap:wrap;
            gap:6px;
            font-size: 11px;
            color:#4b5563;
        }

        /* Global button styling for this tab */
        div[data-testid="stButton"] > button {
            border-radius:999px;
            padding:0.35rem 0.95rem;
            border:1px solid #e5e7eb;
            background:linear-gradient(135deg,#f9fafb,#eef2ff);
            color:#111827;
            font-size:0.85rem;
        }
        div[data-testid="stButton"] > button:hover {
            border-color:#6366f1;
            background:linear-gradient(135deg,#eef2ff,#e0f2fe);
            box-shadow:0 0 0 1px rgba(99,102,241,0.25);
        }

        .ai-q-sub {
            font-size:11px;
            color:#6b7280;
            margin-top:4px;
            min-height:2.2em;
        }
        .ai-view-suggestion {
            margin-top:12px;
            padding:10px 12px;
            border-radius:9px;
            border:1px dashed #d4d4d8;
            background:#fafafa;
            font-size:12px;
        }
        .ai-view-reason {
            color:#4b5563;
        }
        .ai-view-note {
            color:#9ca3af;
        }

        /* Review pills */
        .ai-review-bar {
            display:flex;
            flex-wrap:wrap;
            gap:10px;
            margin: 10px 0 14px 0;
        }
        .ai-review-pill {
            min-width:220px;
            padding:10px 12px;
            border-radius:12px;
            background:linear-gradient(135deg,#fff7ed,#fefce8);
            border:1px solid #fed7aa;
            box-shadow:0 4px 8px rgba(248, 171, 104, 0.15);
            font-size:11px;
        }
        .ai-review-header {
            display:flex;
            align-items:center;
            gap:6px;
            margin-bottom:3px;
        }
        .ai-review-badge {
            padding:2px 7px;
            border-radius:999px;
            font-size:9px;
            text-transform:uppercase;
            letter-spacing:0.03em;
            color:#fff;
        }
        .ai-review-badge-mecca {
            background:#0f766e;
        }
        .ai-review-badge-comp {
            background:#4b5563;
        }
        .ai-review-store {
            font-weight:600;
            color:#111827;
        }
        .ai-review-main {
            font-size:12px;
            font-weight:600;
            margin-bottom:1px;
        }
        .ai-review-extra {
            font-size:11px;
            color:#6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("🤖 Mecca AI Analyst")
    st.caption(
        "Ask questions about sales, forecasting, rostering and what customers are saying, "
        "or use one-click cards to show why Mecca’s forecasting engine and customer "
        "experience matter."
    )

    render_data_pills()

    tab_reviews, tab_quick, tab_chat = st.tabs(
        ["🗣️ Customer review intelligence", "⚡ Quick Insights", "💬 Conversational AI"]
    )

    with tab_reviews:
        render_review_intelligence()

    with tab_quick:
        render_quick_insights()

    with tab_chat:
        render_conversational_ai()
