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

st.set_page_config(page_title="Capstone", layout="wide")
st.title("Analytics Dashboard")

# === Constants: reuse the same Static folder ===
DEFAULT_STATIC = os.path.join(os.path.dirname(__file__), "Static")
STATIC_DIR = st.session_state.get("storage_paths", {}).get("STATIC_DIR", DEFAULT_STATIC)

# Single source of truth for the current handle on this page
CURRENT_USER = (
    st.session_state.get("last_username")
    or st.session_state.get("username", "")
).strip()

RUNS = st.session_state.get("scrape_state", {}).get("runs", {})

if RUNS:
    available = sorted(RUNS.keys())
else:
    # show a message and stop
    if "list_available_users" not in globals():
        st.info("No cached data found. Please scrape some users first.")
        st.stop()
    try:
        available = list_available_users(STATIC_DIR)  
    except NameError:
        st.info("No cached data found. Please scrape some users first.")
        st.stop()
    except Exception:
        st.info("No cached data found. Please scrape some users first.")
        st.stop()

# === helpers to load & render the header ===
def list_available_users(static_dir: str):
    """List usernames that have saved header + post metadata from the scraper page."""
    if not os.path.isdir(static_dir):
        return []
    users = []
    for name in os.listdir(static_dir):
        user_dir = os.path.join(static_dir, name)
        has_profile = os.path.exists(os.path.join(user_dir, "profile.json"))
        has_meta    = os.path.exists(os.path.join(user_dir, "metadata.json"))
        if os.path.isdir(user_dir) and has_profile and has_meta:
            users.append(name)
    return sorted(users, key=str.lower)

def load_profile_bundle(username: str):
    """Read profile.json and find profile.jpg path (if present)."""
    user_dir = os.path.join(STATIC_DIR, username)
    profile_json_path = os.path.join(user_dir, "profile.json")
    profile_pic_path  = os.path.join(user_dir, "profile.jpg")
    if not os.path.exists(profile_json_path):
        raise FileNotFoundError(f"No profile.json for {username} in {user_dir}")
    with open(profile_json_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    if not os.path.exists(profile_pic_path):
        profile_pic_path = None
    return profile, profile_pic_path

def render_profile_header(profile: dict, profile_pic_path: str|None):
    """Header box centered horizontally; internal elements unchanged."""
    # OUTER: 3-column frame to center the header box
    _left, mid, _right = st.columns([1, 2, 1])  # adjust middle ratio to make box wider/narrower
    with mid:
        with st.container(border=False):  # the visible header box
            # padding to match design
            st.markdown('<div style="padding: 12px 16px;">', unsafe_allow_html=True)

            # INNER: two-column layout
            c1, c2 = st.columns([1, 4], vertical_alignment="center")
            with c1:
                if profile_pic_path:
                    st.image(profile_pic_path, width=96)
            with c2:
                st.markdown(f"### {profile['username']}")
                st.markdown(
                    f"**{profile['mediacount']:,}** posts &nbsp;&nbsp; "
                    f"**{profile['followers']:,}** followers &nbsp;&nbsp; "
                    f"**{profile['followees']:,}** following"
                )
                if profile.get("biography"):
                    st.caption(profile["biography"]) 

            st.markdown('</div>', unsafe_allow_html=True)

    # separator under the header box
    st.markdown("---")


# === helper to load saved per-post metadata for heatmap ====================
def load_meta_records(username: str):
    """Load the per-post metadata your scraper saved: Static/<username>/metadata.json."""
    user_dir = os.path.join(STATIC_DIR, username)
    meta_path = os.path.join(user_dir, "metadata.json")
    if not os.path.exists(meta_path):
        # Return empty list instead of raising so UI can show hint
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records

# === convert metadata -> calendar matrix (rows Mon..Sun, cols week starts) ===
def build_calendar_matrix_from_meta(meta_records, year: int):
    """
    Convert meta_out (list of posts) → 7xN matrix for a calendar heatmap.
    - Parse 'taken_at' as UTC and use the UTC calendar date (no timezone selection).
    - Filter by year and aggregate counts per day.
    """
    if not meta_records:
        return pd.DataFrame()

    df = pd.DataFrame(meta_records)
    if df.empty or "taken_at" not in df.columns:
        return pd.DataFrame()

    # Parse as UTC and use the UTC date directly
    dt_utc = pd.to_datetime(df["taken_at"], errors="coerce", utc=True)
    df["date"] = dt_utc.dt.date

    # Keep only selected year
    mask = pd.to_datetime(df["date"]).dt.year == year
    daily = (
        pd.DataFrame({"date": pd.to_datetime(df.loc[mask, "date"])})
        .groupby("date").size().rename("count").reset_index()
    )

    # Build full-year calendar and fill missing days with 0
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    cal = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    if not daily.empty:
        daily["date"] = daily["date"].dt.tz_localize(None)
    cal = cal.merge(daily, on="date", how="left").fillna({"count": 0})
    cal["count"] = cal["count"].astype(int)


    # Rows = Mon..Sun (0..6); Cols = week starts (Mondays)
    cal["dow"] = cal["date"].dt.weekday
    cal["week_start"] = (cal["date"] - pd.to_timedelta(cal["dow"], unit="D")).dt.date
    pivot = cal.pivot_table(index="dow", columns="week_start", values="count", aggfunc="sum", fill_value=0)
    return pivot.reindex(index=range(0, 7))

# ===  Plotly calendar heatmap ============================
def calendar_heatmap_figure(pivot: pd.DataFrame, cell_px: int = 22, gaps=(1, 1)) -> go.Figure:
    if pivot.empty:
        return go.Figure()

    x = [pd.to_datetime(str(d)) for d in pivot.columns]  # week starts (Mondays)
    y_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    z = pivot.values

    # Full-year month ticks
    year = pd.to_datetime(str(pivot.columns[0])).year
    x_range_start = pd.Timestamp(year=year, month=1, day=1)
    x_range_end   = pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=1)

    # Hover text
    hover_text = []
    for r, dow in enumerate(range(7)):
        row = []
        for c, week_start in enumerate(pivot.columns):
            day = pd.to_datetime(str(week_start)) + pd.Timedelta(days=dow)
            row.append(f"{day.date()}: {z[r][c]} post(s)")
        hover_text.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y_labels,
        text=hover_text, hoverinfo="text",
        coloraxis="coloraxis",
        xgap=gaps[0], ygap=gaps[1],   # set once here
    ))

    # Compute a height so cells are roughly square
    ny = pivot.shape[0]  # 7
    fig.update_layout(
        margin=dict(l=50, r=70, t=40, b=40),
        height=ny * cell_px + 80,                 # <- square-ish rows
        coloraxis=dict(colorscale="Greens", colorbar=dict(x=1.02)),
        xaxis=dict(type="date", range=[x_range_start, x_range_end],
                   dtick="M1", tickformat="%b", ticks="outside", showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig

## === Pick which profile(s) to show — USE IN-MEMORY CACHE ===
RUNS = st.session_state.get("scrape_state", {}).get("runs", {})

if not RUNS:
    st.info(
        "No cached profiles yet. Go to the Scraper page and click **Download / Update Posts**."
    )
    st.stop()

available = sorted(RUNS.keys())

# put last-used first so its tab opens first
default_user = st.session_state.get("last_username")
if default_user and default_user in available:
    available = [default_user] + [u for u in available if u != default_user]

# === Tabs row + compact year dropdown (same row)
st.markdown("### Accounts")

tabs = st.tabs(available)

current_year = int(pd.Timestamp.now().year)
years = list(range(current_year, current_year - 6, -1))

year = st.selectbox(
    "Calendar year",
    options=years,
    index=0,
    key="analytics_year_main",
    help="Select the year to display posts for.",
)

# default tab index based on last used user
idx = 0
default_user = st.session_state.get("last_username")
if default_user and default_user in available:
    idx = available.index(default_user)

# === render each account inside the tab loop
for tab, username in zip(tabs, available):
    with tab:
        st.session_state["username"] = username  # keep in sync for other pages

        # Load posts/meta for this user
        # Prefer in-memory cache from the Scraper with fall back to disk
        if username in RUNS:
            meta_records = RUNS[username].get("posts", [])
        else:
            meta_records = load_meta_records(username)

# 1) Profile header OUTSIDE the bordered card
# === Render analytics for each profile via tab navigation ==================
for tab, username in zip(tabs, available):
    with tab:
        st.session_state["username"] = username  # keep in sync for other pages

        # Build profile dict from cache for the header
        _d = RUNS.get(username, {})
        profile_dict = {
            "username": _d.get("username", username),
            "biography": _d.get("biography", ""),
            "followers": _d.get("followers", 0),
            "followees": _d.get("followees", 0),
            "mediacount": _d.get("mediacount", 0),
        }
        pic_path = _d.get("profile_pic_path")

        try:
            render_profile_header(profile_dict, pic_path)
        except Exception as e:
            st.error(f"Could not render header for {username or '<unknown>'}: {e}")
            continue

        # Prefer in-memory posts
        meta_records = _d.get("posts", [])
        
        if not meta_records:
            st.warning(
                "No posts found for this profile."
            )
            continue

        st.caption(f"Calendar year: {year}")

        # Account Overview, Heatmap, Share of Sentiment, Breakdown, Line Over Time, Table
        try:
            pivot = build_calendar_matrix_from_meta(meta_records, int(year))
            if pivot.empty:
                st.info("No posts for the selected year (or metadata has no timestamps).")
            else:
                pass
        except Exception as e:
            st.warning(f"Heatmap unavailable: {e}")

        try:
            pivot = build_calendar_matrix_from_meta(meta_records, int(year))
            if pivot.empty:
                st.info("No posts for the selected year (or metadata has no timestamps).")
            else:
                # 2×3 layout
                r1c1, r1c2, r1c3 = st.columns(3, gap="small")  # <- donut gets more room

                CARD_H = 360                                    # default card height (320–400 work well)
                r2c1, r2c2, r2c3 = st.columns(3, gap="small")   # row 2

                # === ROW 1 ===
                # TOP-LEFT — Account Overview/Summary card
                with r1c1:
                    with st.container(border=True):
                        st.markdown("#### Account Overview")
                        st.metric("Total posts", f"{int(pivot.values.sum()):,}")
                        st.metric("Active days", f"{int((pivot.values>0).sum()):,}")
                        st.metric("Max/day", f"{int(pivot.values.max()) if pivot.size else 0:,}")

                # TOP-MIDDLE - Heatmap
                with r1c2:
                    with st.container(border=True):
                        st.markdown("#### Posts per Day")
                        fig = calendar_heatmap_figure(pivot, cell_px=26, gaps=(1, 1)) # Adjusts cell width and border gaps
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True}, key=f"heatmap_{username}_{year}")


                # TOP-RIGHT - Pie Chart
                with r1c3:
                    with st.container(border=True):
                        st.markdown("#### Share of Sentiment")

                        df = pd.DataFrame(meta_records)

                        if df.empty:
                            st.info("No posts loaded.")
                        else:
                            # --- labels ---
                            if "sentiment_label" in df.columns and df["sentiment_label"].notna().any():
                                lab = df["sentiment_label"].astype(str).str.lower().str.strip()
                            elif "sentiment_score" in df.columns:
                                s = df["sentiment_score"]
                                lab = pd.Series(pd.NA, index=df.index)
                                lab = lab.mask(s >= 0.60, "positive").mask(s <= 0.40, "negative").fillna("neutral")
                            else:
                                st.info("Need either 'sentiment_label' or 'sentiment_score' to plot sentiment share.")
                                st.stop()

                            lab = lab.map({"positive":"Positive","neutral":"Neutral","negative":"Negative"})
                            counts = lab.value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0).astype(int)

                            if counts.sum() == 0:
                                st.info("No sentiment-labelled posts to display.")
                            else:
                                # === CHART DIMENSIONS ===
                                CARD_H = 260                 # overall chart height
                                DONUT_HOLE = 0.55            # 0 (pie) ... 0.9 (thin ring)
                                DOMAIN_X = [0.0, 0.99]       # leave room on right for legend (0..1)

                                fig = go.Figure(go.Pie(
                                    labels=counts.index.tolist(),
                                    values=counts.values.tolist(),
                                    hole=DONUT_HOLE,
                                    textinfo="percent",
                                    sort=False,
                                    marker=dict(colors=["#21A366", "#F1C40F", "#E74C3C"]),
                                    domain=dict(x=DOMAIN_X, y=[0, 1]),   # <- controls donut width inside figure
                                ))
                                fig.update_traces(textfont_size=14)

                                fig.update_layout(
                                    height=CARD_H,            # <- figure height
                                    legend=dict(
                                        orientation="v",
                                        x=1.02, xanchor="left",
                                        y=0.5,  yanchor="middle",
                                        bgcolor="rgba(255,255,255,0.9)",   # legend background 
                                        bordercolor="rgba(0,0,0,0.25)",    # border color
                                        borderwidth=1                      # border thickness (px)
                                    ),
                                    margin=dict(l=10, r=120, t=10, b=10)   # give a bit more space if needed
                                )
                                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"sentiment_pie_{username}")


                # === ROW 2 ====
                # BOTTOM-LEFT — Sentiment Breakdown
                with r2c1:
                    with st.container(border=True):
                        st.markdown("#### Sentiment Breakdown")



                        df = pd.DataFrame(meta_records)
                        if df.empty:
                            st.info("No metadata loaded.")
                        else:
                            # --- timestamp column & parse ---
                            ts_col = "taken_at" if "taken_at" in df.columns else ("date_utc" if "date_utc" in df.columns else None)
                            if ts_col is None:
                                st.info("Missing 'taken_at' or 'date_utc' in metadata.")
                            else:
                                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

                                # --- sentiment labels (or bin score) ---
                                if "sentiment_label" in df.columns and df["sentiment_label"].notna().any():
                                    lab = df["sentiment_label"].astype(str).str.lower().str.strip()
                                elif "sentiment_score" in df.columns:
                                    s = df["sentiment_score"]
                                    lab = pd.Series(pd.NA, index=df.index)
                                    lab = lab.mask(s >= 0.60, "positive").mask(s <= 0.40, "negative").fillna("neutral")
                                else:
                                    st.info("Need 'sentiment_label' or 'sentiment_score'.")
                                    st.stop()

                                df["sent"] = lab.map({"positive":"Positive","neutral":"Neutral","negative":"Negative"})
                                df = df.dropna(subset=[ts_col, "sent"]).copy()

                                # --- build a continuous list of last K months (K = max(5, #months present)) ---
                                df["period"] = df[ts_col].dt.to_period("M")          # e.g., 2025-09
                                if df["period"].empty:
                                    st.info("No dated posts to summarise.")
                                    st.stop()

                                months_present = sorted(df["period"].unique())
                                last_p = df["period"].max()
                                K = max(5, len(months_present))
                                months_seq = [last_p - i for i in range(K)][::-1]    # oldest..newest continuous sequence

                                # --- aggregate counts per (period × sentiment) ---
                                counts = (df.groupby(["period","sent"])
                                            .size().rename("count").reset_index())

                                classes = ["Positive","Neutral","Negative"]
                                idx = pd.MultiIndex.from_product([months_seq, classes], names=["period","sent"])
                                counts = (counts.set_index(["period","sent"])
                                                .reindex(idx, fill_value=0)
                                                .reset_index())

                                # month names (no year) for y labels, in order
                                counts["month_name"] = counts["period"].dt.strftime("%b")
                                order_names = [p.strftime("%b") for p in months_seq]
                                counts["month_name"] = pd.Categorical(counts["month_name"], categories=order_names, ordered=True)
                                counts = counts.sort_values(["month_name","sent"])

                                # ---- x-axis every 10; range up to nearest 10 ----
                                totals = counts.groupby("month_name")["count"].sum()
                                xmax = int(math.ceil((totals.max() if len(totals) else 0)/10.0) * 10) or 10

                                # --- plot: stacked horizontal bars ---
                                fig = px.bar(
                                    counts, y="month_name", x="count",
                                    color="sent", orientation="h", barmode="stack",
                                    labels={"month_name":"Month", "count":"No. of Posts", "sent":""},
                                    color_discrete_map={"Positive":"#21A366","Neutral":"#F1C40F","Negative":"#E74C3C"},
                                )
                                fig.update_xaxes(tickmode="linear", dtick=10, range=[0, xmax])
                                fig.update_layout(
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                    legend=dict(orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"),
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"sentiment_breakdown_{username}")

                # BOTTOM-MIDDLE - Line Graph
                with r2c2:
                    with st.container(border=True):
                        st.markdown("#### Sentiment Over Time")


                        df = pd.DataFrame(meta_records)
                        if df.empty:
                            st.info("No sentiment data available.")
                        else:
                            # pick timestamp & parse
                            ts_col = "taken_at" if "taken_at" in df.columns else ("date_utc" if "date_utc" in df.columns else None)
                            if ts_col is None:
                                st.info("Need 'taken_at' or 'date_utc' in metadata."); st.stop()
                            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

                            # labels 
                            if "sentiment_label" in df.columns and df["sentiment_label"].notna().any():
                                lab = df["sentiment_label"].astype(str).str.lower().str.strip()
                            elif "sentiment_score" in df.columns:
                                s = df["sentiment_score"]
                                lab = pd.Series(pd.NA, index=df.index)
                                lab = lab.mask(s >= 0.60, "positive").mask(s <= 0.40, "negative").fillna("neutral")
                            else:
                                st.info("Need 'sentiment_label' or 'sentiment_score'."); st.stop()

                            df = df.dropna(subset=[ts_col]).copy()
                            df["day"]  = df[ts_col].dt.date
                            df["sent"] = lab.map({"positive":"Positive","neutral":"Neutral","negative":"Negative"})

                            counts = (df.groupby(["day","sent"]).size()
                                        .rename("count").reset_index())

                            classes = ["Positive","Neutral","Negative"]
                            color_map = {"Positive":"#21A366","Neutral":"#F1C40F","Negative":"#E74C3C"}

                            fig = go.Figure()
                            classes = ["Positive","Neutral","Negative"]
                            color_map = {"Positive":"#21A366","Neutral":"#F1C40F","Negative":"#E74C3C"}

                            for cls in classes:
                                sub = counts[counts["sent"] == cls].sort_values("day")

                                fig.add_trace(go.Scatter(
                                    x=sub["day"], y=sub["count"], name=cls,
                                    mode="lines+markers",
                                    line=dict(width=2, color=color_map[cls]),
                                    marker=dict(size=6),
                                    line_shape="spline",  # smooth curve
                                    hovertemplate=f"{cls}<br>%{{x}}<br>No. of posts: %{{y}}<extra></extra>"
                                ))


                            fig.update_layout(
                                legend=dict(
                                    orientation="h",
                                    x=0.5, xanchor="center",   # center horizontally
                                    y=-0.30, yanchor="top"     # place below the chart (under date )
                                ),
                                margin=dict(l=10, r=10, t=10, b=120)  # give enough bottom room so it isn't clipped
                            )

                            fig.update_xaxes(title_text="Date")
                            fig.update_yaxes(title_text="No. of Posts", rangemode="tozero")

                            st.plotly_chart(fig, use_container_width=True, key=f"sentiment_over_time_{username}")
    
                # BOTTOM-RIGHT — Post Table
                with r2c3:
                    with st.container(border=True):
                        st.markdown("#### Posts · Sentiment & Risk [PLACEHOLDER]")



                        df = pd.DataFrame(meta_records)
                        if df.empty:
                            st.info("No metadata loaded.")
                        else:
                            # pick a timestamp column and parse
                            ts_col = "taken_at" if "taken_at" in df.columns else ("date_utc" if "date_utc" in df.columns else None)
                            if ts_col is None:
                                st.info("Missing 'taken_at' or 'date_utc' in metadata.")
                            else:
                                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

                                # --- Compute a basic risk score from sentiment --- (Align with Jonty's)
                                # If score exists: 100 * (1 - score). Otherwise map label.
                                def risk_from_row(r):
                                    if pd.notna(r.get("sentiment_score")):
                                        try:
                                            return int(round(100 * (1.0 - float(r["sentiment_score"]))))
                                        except Exception:
                                            pass
                                    lab = str(r.get("sentiment_label", "")).lower()
                                    return {"negative": 90, "neutral": 50, "positive": 10}.get(lab, 50)

                                df["risk_score"] = df.apply(risk_from_row, axis=1)

                                # Build table
                                table = pd.DataFrame({
                                    "Post": df.get("shortcode", pd.Series(index=df.index)).fillna(""),
                                    "Date": df[ts_col].dt.strftime("%Y-%m-%d"),
                                    "Image sentiment": df.get("prediction", pd.Series(index=df.index)).fillna(""),
                                    "Text sentiment": df.get("sentiment_label", pd.Series(index=df.index)).fillna(""),
                                    "Risk score": df["risk_score"],
                                })

                                # Sort by risk (highest first) and show top N
                                N = st.slider("Rows to display", 5, 100, min(20, len(table)), key=f"post_table_rows_{username}")
                                table = table.sort_values("Risk score", ascending=False).head(N)

                                st.dataframe(
                                    table,
                                    use_container_width=True,
                                    hide_index=True,
                                )

        except Exception as e:
            st.warning(f"Heatmap unavailable: {e}")









