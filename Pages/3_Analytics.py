import streamlit as st
import calendar
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import json
from dateutil import tz
import math
import plotly.express as px
from functools import lru_cache
from Pages.Utilities.utils_analytics import *
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math

# Import all constants and helper utilities
from Pages.Utilities.utils_analytics import (
    STATIC_DIR, COLOR_MAP, TABLE_STYLES,
    list_available_users, load_meta_records, render_profile_header,
    build_calendar_matrix_from_meta, calendar_heatmap_figure,
    pick_timestamp_col, to_year_df, sentiment_series, label_from_prediction_cell
)

# ============================================================================
# App setup
# ============================================================================
st.set_page_config(page_title="Capstone", layout="wide")
st.title("Analytics Dashboard")

# ============================================================================
# Data availability
# ============================================================================
RUNS = st.session_state.get("scrape_state", {}).get("runs", {})

if not RUNS:
    try:
        available = list_available_users(STATIC_DIR)
    except Exception:
        st.info("No cached data found. Please scrape some users first.")
        st.stop()
    if not available:
        st.info("No cached profiles yet. Go to the Scraper page and click **Download / Update Posts**.")
        st.stop()
else:
    available = sorted(RUNS.keys())

# Default to last used
default_user = st.session_state.get("last_username")
if default_user and default_user in available:
    available = [default_user] + [u for u in available if u != default_user]

st.markdown("### Accounts")
tabs = st.tabs(available)

# ============================================================================
# Render each account tab
# ============================================================================
for tab, username in zip(tabs, available):
    with tab:
        st.session_state["username"] = username

        # --- Profile header ---
        cached = RUNS.get(username, {})
        profile = {
            "username": cached.get("username", username),
            "biography": cached.get("biography", ""),
            "followers": cached.get("followers", 0),
            "followees": cached.get("followees", 0),
            "mediacount": cached.get("mediacount", 0),
        }
        pic_path = cached.get("profile_pic_path")
        render_profile_header(profile, pic_path)

        # --- Metadata load ---
        meta_records = cached.get("posts", []) or load_meta_records(username)
        if not meta_records:
            st.warning("No posts found for this profile.")
            continue

        # --- Year selection ---
        df_all = pd.DataFrame(meta_records)
        ts_col = pick_timestamp_col(df_all)
        if ts_col is None:
            st.warning("No 'taken_at' or 'date_utc' field found in data.")
            st.stop()
        df_all[ts_col] = pd.to_datetime(df_all[ts_col], errors="coerce", utc=True)
        years = sorted(df_all[ts_col].dt.year.dropna().unique(), reverse=True)
        selected_year = st.selectbox("Select Calendar Year", years, index=0, key=f"year_{username}")
        year = int(selected_year)

        # --- Shared data for this tab/year ---
        df_year = to_year_df(meta_records, year)
        pivot = build_calendar_matrix_from_meta(meta_records, year)

        # Layout
        r1c1, r1c2, r1c3 = st.columns(3, gap="small")
        r2c1, r2c2, r2c3 = st.columns(3, gap="small")

        # =========================================================================
        # ROW 1, CARD 1 — Account Overview
        # =========================================================================
        with r1c1:
            with st.container(border=True):
                st.markdown("#### Account Overview")
                if df_year.empty or pivot.empty:
                    st.info("No data available for the selected year.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Total posts", f"{int(pivot.values.sum()):,}")
                        st.metric("Active days", f"{int((pivot.values > 0).sum()):,}")
                        st.metric("Max/day", f"{int(pivot.values.max()) if pivot.size else 0:,}")

                    with c2:
                        if "risk_score" in df_year.columns:
                            df_year["risk_score"] = pd.to_numeric(df_year["risk_score"], errors="coerce")
                            avg_risk = df_year["risk_score"].mean()
                            avg_risk_txt = f"{0 if pd.isna(avg_risk) else avg_risk:.1f}"

                            df_year["risk_band"] = pd.cut(
                                df_year["risk_score"],
                                bins=[0, 40, 70, 100],
                                labels=["Low", "Medium", "High"],
                                include_lowest=True,
                            )
                            low_risk_count = int((df_year["risk_band"] == "Low").sum())
                            high_risk_count = int((df_year["risk_band"] == "High").sum())
                        else:
                            avg_risk_txt = "0.0"
                            low_risk_count = 0
                            high_risk_count = 0

                        st.metric("Low Risk", f"{low_risk_count:,}")
                        st.metric("High Risk", f"{high_risk_count:,}")
                        st.metric("Risk Score", avg_risk_txt)

        # =========================================================================
        # ROW 1, CARD 2 — Calendar Heatmap
        # =========================================================================
        with r1c2:
            with st.container(border=True):
                st.markdown("#### Posts per Day")
                if pivot.empty:
                    st.info("No posts for the selected year.")
                else:
                    fig = calendar_heatmap_figure(pivot, cell_px=26, gaps=(1, 1))
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": True},
                        key=f"heatmap_{username}_{year}",
                    )

        # =========================================================================
        # ROW 1, CARD 3 — Share of Sentiment Donut Chart
        # =========================================================================
        with r1c3:
            with st.container(border=True):
                st.markdown("#### Share of Sentiment")
                df = pd.DataFrame(meta_records)
                lab = sentiment_series(df)
                if lab.empty or lab.isna().all():
                    st.info("Need either 'sentiment_label' or 'sentiment_score' to plot sentiment share.")
                else:
                    counts = (
                        lab.value_counts()
                        .reindex(["Positive", "Neutral", "Negative"])
                        .fillna(0)
                        .astype(int)
                    )
                    if counts.sum() == 0:
                        st.info("No sentiment-labelled posts to display.")
                    else:
                        fig = go.Figure(
                            go.Pie(
                                labels=counts.index.tolist(),
                                values=counts.values.tolist(),
                                hole=0.55,
                                textinfo="percent",
                                sort=False,
                                marker=dict(colors=[
                                    COLOR_MAP["Positive"],
                                    COLOR_MAP["Neutral"],
                                    COLOR_MAP["Negative"],
                                ]),
                                domain=dict(x=[0.0, 0.99], y=[0, 1]),
                            )
                        )
                        fig.update_layout(
                            height=260,
                            legend=dict(
                                orientation="v",
                                x=1.02, xanchor="left",
                                y=0.5, yanchor="middle",
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="rgba(0,0,0,0.25)",
                                borderwidth=1,
                            ),
                            margin=dict(l=10, r=120, t=10, b=10),
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"sentiment_pie_{username}",
                        )

        # =========================================================================
        # ROW 2, CARD 1 — Sentiment Breakdown Bar Graph
        # =========================================================================
        with r2c1:
            with st.container(border=True):
                st.markdown("#### Sentiment Breakdown")
                if df_year.empty:
                    st.info("No metadata loaded for the selected year.")
                else:
                    ts_col = pick_timestamp_col(df_year)
                    df_year[ts_col] = pd.to_datetime(df_year[ts_col], errors="coerce", utc=True)
                    lab = sentiment_series(df_year)
                    df_year["sent"] = lab
                    df_year["period"] = df_year[ts_col].dt.to_period("M")

                    counts = (
                        df_year.groupby(["period", "sent"]).size().rename("count").reset_index()
                    )
                    months_seq = pd.period_range(df_year["period"].min(), df_year["period"].max(), freq="M")
                    classes = ["Positive", "Neutral", "Negative"]
                    idx = pd.MultiIndex.from_product([months_seq, classes], names=["period", "sent"])
                    counts = (
                        counts.set_index(["period", "sent"]).reindex(idx, fill_value=0).reset_index()
                    )

                    counts["month_label"] = (
                        counts["period"].dt.to_timestamp().dt.strftime("%b %Y")
                    )
                    order_labels = [p.to_timestamp().strftime("%b %Y") for p in months_seq]
                    counts["month_label"] = pd.Categorical(
                        counts["month_label"], categories=order_labels, ordered=True
                    )
                    counts = counts.sort_values(["month_label", "sent"])

                    totals = counts.groupby("month_label")["count"].sum()
                    xmax = int(math.ceil((totals.max() if len(totals) else 0) / 10.0) * 10) or 10

                    fig = px.bar(
                        counts,
                        y="month_label",
                        x="count",
                        color="sent",
                        orientation="h",
                        barmode="stack",
                        labels={"month_label": "Month", "count": "No. of Posts", "sent": ""},
                        color_discrete_map=COLOR_MAP,
                    )
                    fig.update_yaxes(categoryorder="array", categoryarray=order_labels)
                    fig.update_xaxes(tickmode="linear", dtick=10, range=[0, xmax])
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"sentiment_breakdown_{username}_{selected_year}",
                    )

        # =========================================================================
        # ROW 2, CARD 2 — Sentiment Over Time LineGraph
        # =========================================================================
        with r2c2:
            with st.container(border=True):
                st.markdown("#### Sentiment Over Time")
                if df_year.empty:
                    st.info("No sentiment data available for the selected year.")
                else:
                    ts_col = pick_timestamp_col(df_year)
                    df_y = df_year.copy()
                    df_y[ts_col] = pd.to_datetime(df_y[ts_col], errors="coerce", utc=True)
                    lab = sentiment_series(df_y)
                    df_y["day"] = df_y[ts_col].dt.date
                    df_y["sent"] = lab

                    counts = (
                        df_y.groupby(["day", "sent"]).size().rename("count").reset_index()
                    )

                    fig = go.Figure()
                    for cls in ["Positive", "Neutral", "Negative"]:
                        sub = counts[counts["sent"] == cls].sort_values("day")
                        fig.add_trace(
                            go.Scatter(
                                x=sub["day"], y=sub["count"],
                                name=cls, mode="lines+markers",
                                line=dict(width=2, color=COLOR_MAP[cls]),
                                marker=dict(size=6),
                                line_shape="spline",
                                hovertemplate=f"{cls}<br>%{{x}}<br>No. of posts: %{{y}}<extra></extra>",
                            )
                        )

                    fig.update_layout(
                        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.30, yanchor="top"),
                        margin=dict(l=10, r=10, t=10, b=120),
                    )
                    fig.update_xaxes(tickformat="%b %d", hoverformat="%b %d, %Y")
                    fig.update_yaxes(tickmode="linear", dtick=1, tick0=0, rangemode="tozero")
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"sentiment_over_time_{username}_{selected_year}",
                    )

        # =========================================================================
        # ROW 2, CARD 3 — Posts Table
        # =========================================================================
        with r2c3:
            with st.container(border=True):
                st.markdown("#### Posts · Sentiment & Risk")
                df = pd.DataFrame(meta_records)
                ts_col = pick_timestamp_col(df)
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

                df["risk_score"] = (
                    pd.to_numeric(df.get("risk_score", pd.Series(index=df.index)), errors="coerce")
                    .round()
                    .astype("Int64")
                    .fillna(0)
                    .astype(int)
                )
                df["risk_band"] = df.get("risk_band", "").astype(str).str.upper().str.strip()

                date_fmt = "%b %d, %Y"
                img_label = (
                    df.get("prediction", pd.Series(index=df.index))
                    .apply(label_from_prediction_cell)
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.capitalize()
                )
                text_sent = (
                    df.get("sentiment_label", pd.Series(index=df.index))
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map({"POSITIVE": "Positive", "NEUTRAL": "Neutral", "NEGATIVE": "Negative"})
                    .fillna("")
                )
                table = pd.DataFrame({
                    "Date": df[ts_col].dt.strftime(date_fmt),
                    "Image sentiment": img_label,
                    "Text sentiment": text_sent,
                    "Risk score": df["risk_score"],
                    "Post ID": df.get("shortcode", pd.Series(index=df.index)).fillna(""),
                })

                N = min(100, len(table))
                table = table.sort_values("Risk score", ascending=False).head(N)
                table.index = np.arange(1, len(table) + 1)
                table.index.name = "Row"
                table = table.reset_index()

                SENT_GREEN = "#1F7A3E"
                SENT_RED = "#B3392F"
                SENT_AMBER = "#8A6D3B"

                def style_sentiment_color(col: pd.Series) -> list[str]:
                    css = []
                    for v in col.astype(str):
                        if v == "Positive":
                            css.append(f"color:{SENT_GREEN}; font-weight:600;")
                        elif v == "Negative":
                            css.append(f"color:{SENT_RED}; font-weight:600;")
                        elif v == "Neutral":
                            css.append(f"color:{SENT_AMBER}; font-weight:600;")
                        else:
                            css.append("")
                    return css

                styled = (
                    table.style.set_table_styles(TABLE_STYLES)
                    .hide(axis="index")
                    .set_properties(
                        subset=table.columns,
                        **{
                            "white-space": "nowrap",
                            "text-align": "center",
                            "font-family": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                        },
                    )
                    .apply(style_sentiment_color, subset=["Image sentiment", "Text sentiment"])
                )

                import streamlit.components.v1 as components
                CARD_MAX_HEIGHT = 420
                html = f"""
                <style>
                html, body {{
                    background: transparent;
                    margin: 0; padding: 0;
                    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
                }}
                .wrapper {{
                    max-height: {CARD_MAX_HEIGHT}px;
                    overflow-y: auto;
                    overflow-x: auto;
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 10px;
                    background: transparent;
                }}
                table {{ width: 100%; border-collapse: separate; border-spacing: 0; }}
                thead th {{ position: sticky; top: 0; z-index: 1; background: #171d26; font-family: inherit; }}
                tbody td, tbody th {{ font-family: inherit; }}
                </style>
                <div class="wrapper">{styled.to_html()}</div>
                """
                components.html(html, height=CARD_MAX_HEIGHT + 24, scrolling=False)







