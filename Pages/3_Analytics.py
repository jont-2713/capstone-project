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
#

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

    # --- NEW: compute the occupied x-window and pad it ---
    # columns with any nonzero count
    nonzero_weeks = pivot.columns[(pivot > 0).any(axis=0)]

    # overall domain of your calendar (min/max week columns you plotted)
    domain_start = pd.to_datetime(str(pivot.columns.min()))
    # add one week so the last column is fully included
    domain_end   = pd.to_datetime(str(pivot.columns.max())) + pd.Timedelta(weeks=1)

    if len(nonzero_weeks) > 0:
        dmin = pd.to_datetime(str(nonzero_weeks.min()))
        dmax = pd.to_datetime(str(nonzero_weeks.max()))

        pad  = pd.Timedelta(days=7)     # small visual margin
        six  = pd.Timedelta(days=183)   # ~ six months

        span = dmax - dmin
        mid  = dmin + span / 2

        # ensure at least six months; otherwise use data span + pad
        half = max(six / 2, span / 2 + pad)

        start = mid - half
        end   = mid + half

        # clamp to calendar domain so we don't scroll off canvas
        start = max(start, domain_start)
        end   = min(end,   domain_end)

        x_range = [start, end]
    else:
        # no activity: default to a centered six-month window within the domain
        mid    = domain_start + (domain_end - domain_start) / 2
        half   = pd.Timedelta(days=183) / 2
        start  = max(domain_start, mid - half)
        end    = min(domain_end,   mid + half)
        x_range = [start, end]

    # Hover text (unchanged)
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
        xgap=gaps[0], ygap=gaps[1],
    ))

    ny = pivot.shape[0]  # 7 rows
    fig.update_layout(
        margin=dict(l=50, r=70, t=40, b=40),
        height=ny * cell_px + 80,
        coloraxis=dict(colorscale="BuGn", colorbar=dict(x=1.02)),
        uirevision=None,                 # avoid restoring stale zoom
        xaxis_constrain="domain",
    )

    # apply the computed range when you update axes (keep your existing call)
    # fig.update_xaxes(range=x_range, dtick="M1", tickformat="%b", ...)


    fig.update_xaxes(
        type="date",
        range=x_range,                 # <- NEW: start zoomed to active weeks
        dtick="M1",
        tickformat="%b",
        ticks="outside",
        showgrid=False
    )
    fig.update_yaxes(showgrid=False)

    # keep light background like before
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
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

# === Tabs row (no year dropdown)
st.markdown("### Accounts")

tabs = st.tabs(available)

# keep a 'year' variable for downstream code, but pick it automatically
year = int(pd.Timestamp.now().year)  # or set from data later per-account


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

        ####
        # --- Year Filter (applies to all charts in this tab) --------------------
        df = pd.DataFrame(meta_records)
        if "taken_at" not in df.columns:
            st.warning("No 'taken_at' field found in data.")
            st.stop()

        df["taken_at"] = pd.to_datetime(df["taken_at"], errors="coerce", utc=True)
        available_years = sorted(df["taken_at"].dt.year.dropna().unique(), reverse=True)

        selected_year = st.selectbox(
            "Select Calendar Year",
            available_years,
            index=0,
            key=f"year_{username}"  # <- per-tab key so user A and user B can have different selections
        )

        # Use this filtered frame for ALL charts/metrics on the tab
        df_year = df[df["taken_at"].dt.year == selected_year].copy()
        year = int(selected_year)  # if your existing code expects `year`


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
                # === ROW 1 ===
                # TOP-LEFT — Account Overview/Summary card
                # --- TOP-LEFT — Account Overview/Summary card ---
                with r1c1:
                    with st.container(border=True):
                        st.markdown("#### Account Overview")

                        # Use filtered data from the selected calendar year
                        if df_year.empty:
                            st.info("No data available for the selected year.")
                        else:
                            # --- Left column: summary metrics from pivot (still year-based) ---
                            c1, c2 = st.columns(2)

                            with c1:
                                st.metric("Total posts", f"{int(pivot.values.sum()):,}")
                                st.metric("Active days", f"{int((pivot.values > 0).sum()):,}")
                                st.metric("Max/day", f"{int(pivot.values.max()) if pivot.size else 0:,}")

                            # --- Right column: risk metrics from df_year ---
                            def _img_label(x):
                                if x is None:
                                    return ""
                                if isinstance(x, dict):
                                    return str(x.get("label", "")).strip()
                                if isinstance(x, str):
                                    return x.strip()
                                if isinstance(x, (list, tuple)):
                                    for item in x:
                                        if isinstance(item, dict) and item.get("label"):
                                            return str(item["label"]).strip()
                                        if isinstance(item, str) and item.strip():
                                            return item.strip()
                                    return ""
                                return str(x).strip()

                            df = df_year.copy()  # use filtered dataset for year
                            img_lab = df.get("prediction", pd.Series(index=df.index)).apply(_img_label).str.capitalize().fillna("")
                            text_lab = df.get("sentiment_label", pd.Series(index=df.index)).astype(str).str.strip().str.capitalize()

                            # --- Risk metrics ---
                            if "risk_score" in df.columns:
                                df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
                                avg_risk = df["risk_score"].mean()
                                avg_risk_txt = f"{0 if pd.isna(avg_risk) else avg_risk:.1f}"

                                # Classify risk bands
                                df["risk_band"] = pd.cut(
                                    df["risk_score"],
                                    bins=[0, 40, 70, 100],
                                    labels=["Low", "Medium", "High"],
                                    include_lowest=True,
                                )

                                low_risk_count = int((df["risk_band"] == "Low").sum())
                                high_risk_count = int((df["risk_band"] == "High").sum())
                            else:
                                avg_risk_txt = "0.0"
                                low_risk_count = 0
                                high_risk_count = 0

                            # --- Right column metrics ---
                            with c2:
                                st.metric("Low Risk", f"{low_risk_count:,}")
                                st.metric("High Risk", f"{high_risk_count:,}")
                                st.metric("Risk Score", avg_risk_txt)




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

                        # Use the year-filtered dataset
                        if df_year.empty:
                            st.info("No metadata loaded for the selected year.")
                        else:
                            ts_col = "taken_at" if "taken_at" in df_year.columns else (
                                "date_utc" if "date_utc" in df_year.columns else None
                            )
                            if ts_col is None:
                                st.info("Missing 'taken_at' or 'date_utc' in metadata.")
                            else:
                                df_year[ts_col] = pd.to_datetime(df_year[ts_col], errors="coerce", utc=True)

                                # --- sentiment labels (or bin score) ---
                                if "sentiment_label" in df_year.columns and df_year["sentiment_label"].notna().any():
                                    lab = df_year["sentiment_label"].astype(str).str.lower().str.strip()
                                elif "sentiment_score" in df_year.columns:
                                    s = df_year["sentiment_score"]
                                    lab = pd.Series(pd.NA, index=df_year.index)
                                    lab = lab.mask(s >= 0.60, "positive").mask(s <= 0.40, "negative").fillna("neutral")
                                else:
                                    st.info("Need 'sentiment_label' or 'sentiment_score'.")
                                    st.stop()

                                df_year["sent"] = lab.map(
                                    {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
                                )
                                df_year = df_year.dropna(subset=[ts_col, "sent"]).copy()

                                # --- build a continuous list of months for the selected year ---
                                df_year["period"] = df_year[ts_col].dt.to_period("M")
                                if df_year["period"].empty:
                                    st.info("No dated posts to summarise.")
                                    st.stop()

                                # months_present = sorted(df_year["period"].unique())
                                # last_p = df_year["period"].max()
                                # K = max(4, len(months_present))
                                # months_seq = [last_p - i for i in range(K)][::-1]

                                # --- show all months chronologically within the selected year ---
                                months_present = sorted(df_year["period"].unique())

                                if not months_present:
                                    st.info("No dated posts to summarise.")
                                    st.stop()

                                # Create a continuous monthly range from the earliest to the latest period in that year
                                start_p, end_p = months_present[0], months_present[-1]
                                months_seq = pd.period_range(start=start_p, end=end_p, freq="M")


                                # --- aggregate counts per (period × sentiment) ---
                                counts = (
                                    df_year.groupby(["period", "sent"])
                                    .size()
                                    .rename("count")
                                    .reset_index()
                                )

                                classes = ["Positive", "Neutral", "Negative"]
                                idx = pd.MultiIndex.from_product([months_seq, classes], names=["period", "sent"])
                                counts = (
                                    counts.set_index(["period", "sent"])
                                    .reindex(idx, fill_value=0)
                                    .reset_index()
                                )

                                # --- unique, ordered month labels ---
                                counts["month_label"] = counts["period"].dt.to_timestamp().dt.strftime("%b %Y")
                                order_labels = [p.to_timestamp().strftime("%b %Y") for p in months_seq]
                                counts["month_label"] = pd.Categorical(
                                    counts["month_label"], categories=order_labels, ordered=True
                                )
                                counts = counts.sort_values(["month_label", "sent"])

                                totals = counts.groupby("month_label")["count"].sum()
                                xmax = int(math.ceil((totals.max() if len(totals) else 0) / 10.0) * 10) or 10

                                # --- plot: stacked horizontal bars ---
                                fig = px.bar(
                                    counts,
                                    y="month_label",
                                    x="count",
                                    color="sent",
                                    orientation="h",
                                    barmode="stack",
                                    labels={"month_label": "Month", "count": "No. of Posts", "sent": ""},
                                    color_discrete_map={
                                        "Positive": "#21A366",
                                        "Neutral": "#F1C40F",
                                        "Negative": "#E74C3C",
                                    },
                                )
                                fig.update_yaxes(categoryorder="array", categoryarray=order_labels)
                                fig.update_xaxes(tickmode="linear", dtick=10, range=[0, xmax])
                                fig.update_layout(
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    legend=dict(
                                        orientation="v", x=1.02, xanchor="left", y=0.5, yanchor="middle"
                                    ),
                                )

                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    key=f"sentiment_breakdown_{username}_{selected_year}",
                                )

                # BOTTOM-MIDDLE - Line Graph
                with r2c2:
                    with st.container(border=True):
                        st.markdown("#### Sentiment Over Time")

                        # Use the year-filtered dataframe
                        if df_year.empty:
                            st.info("No sentiment data available for the selected year.")
                        else:
                            # pick timestamp & parse
                            ts_col = "taken_at" if "taken_at" in df_year.columns else (
                                "date_utc" if "date_utc" in df_year.columns else None
                            )
                            if ts_col is None:
                                st.info("Need 'taken_at' or 'date_utc' in metadata."); st.stop()

                            df_y = df_year.copy()
                            df_y[ts_col] = pd.to_datetime(df_y[ts_col], errors="coerce", utc=True)

                            # labels
                            if "sentiment_label" in df_y.columns and df_y["sentiment_label"].notna().any():
                                lab = df_y["sentiment_label"].astype(str).str.lower().str.strip()
                            elif "sentiment_score" in df_y.columns:
                                s = df_y["sentiment_score"]
                                lab = pd.Series(pd.NA, index=df_y.index)
                                lab = lab.mask(s >= 0.60, "positive").mask(s <= 0.40, "negative").fillna("neutral")
                            else:
                                st.info("Need 'sentiment_label' or 'sentiment_score'."); st.stop()

                            df_y = df_y.dropna(subset=[ts_col]).copy()
                            df_y["day"]  = df_y[ts_col].dt.tz_convert(None).dt.date if hasattr(df_y[ts_col].dt, "tz") else df_y[ts_col].dt.date
                            df_y["sent"] = lab.map({"positive":"Positive","neutral":"Neutral","negative":"Negative"})

                            counts = (
                                df_y.groupby(["day","sent"])
                                    .size()
                                    .rename("count")
                                    .reset_index()
                            )

                            classes = ["Positive","Neutral","Negative"]
                            color_map = {"Positive":"#21A366","Neutral":"#F1C40F","Negative":"#E74C3C"}

                            fig = go.Figure()
                            for cls in classes:
                                sub = counts[counts["sent"] == cls].sort_values("day")
                                fig.add_trace(go.Scatter(
                                    x=sub["day"], y=sub["count"], name=cls,
                                    mode="lines+markers",
                                    line=dict(width=2, color=color_map[cls]),
                                    marker=dict(size=6),
                                    line_shape="spline",
                                    hovertemplate=f"{cls}<br>%{{x}}<br>No. of posts: %{{y}}<extra></extra>"
                                ))

                            fig.update_layout(
                                legend=dict(
                                    orientation="h",
                                    x=0.5, xanchor="center",
                                    y=-0.30, yanchor="top"
                                ),
                                margin=dict(l=10, r=10, t=10, b=120)
                            )

                            # Remove year from X tick labels, keep full date in hover
                            fig.update_xaxes(
                                tickformat="%b %d",
                                hoverformat="%b %d, %Y"
                            )

                            # Whole numbers on Y (counts of posts)
                            fig.update_yaxes(
                                tickmode="linear",
                                dtick=1,
                                tick0=0,
                                rangemode="tozero"
                            )

                            st.plotly_chart(fig, use_container_width=True,
                                            key=f"sentiment_over_time_{username}_{selected_year}")
    
                # BOTTOM-RIGHT — Post Table
                with r2c3:
                    with st.container(border=True):
                        st.markdown("#### Posts · Sentiment & Risk")



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

                            # --- Use risk saved by the scraper (do NOT recompute here) ---
                            df["risk_score"] = (
                                pd.to_numeric(df.get("risk_score", pd.Series(index=df.index)), errors="coerce")
                                .round()
                                .astype("Int64")
                                .fillna(0)
                                .astype(int)
                            )

                            # (optional) also keep the stored band if you want it in tables later
                            if "risk_band" in df.columns:
                                df["risk_band"] = df["risk_band"].astype(str).str.upper().str.strip()




                                # --- Build & style the table (dark theme + sentiment pills) ---

                                # --- Build & style the table (dark theme + sentiment pills) ---

                                # --- Build the table -------------------------------------------------
                                date_fmt = "%b %d, %Y"

                                # label-only image sentiment (capitalised)
                                img_label = (
                                    df.get("prediction", pd.Series(index=df.index))
                                    .apply(lambda x: (x or {}).get("label") if isinstance(x, dict) else "")
                                    .fillna("")
                                    .astype(str).str.strip().str.capitalize()
                                )

                                text_sent = (
                                    df.get("sentiment_label", pd.Series(index=df.index))
                                    .astype(str).str.strip().str.upper()
                                    .map({"POSITIVE": "Positive", "NEUTRAL": "Neutral", "NEGATIVE": "Negative"})
                                    .fillna("")
                                )

                                table = pd.DataFrame({
                                    "Post ID": df.get("shortcode", pd.Series(index=df.index)).fillna(""),
                                    "Date": df[ts_col].dt.strftime(date_fmt),
                                    "Image sentiment": img_label,
                                    "Text sentiment": text_sent,
                                    "Risk score": df["risk_score"],
                                })

                                # --- Limit rows (max 9) & order by risk -----------------------------
                                N = min(100, len(table))
                                table = table.sort_values("Risk score", ascending=False).head(N)

                                # --- Row-number column (make it a REAL column, not index header) ----
                                table.index = np.arange(1, len(table) + 1)   # 1..N
                                table.index.name = "Row"
                                table = table.reset_index()                   # Row becomes a normal column

                                # --- Move Post ID to the END ----------------------------------------
                                table = table[["Row", "Date", "Image sentiment", "Text sentiment", "Risk score", "Post ID"]]

                                # --- Styling: dark table + no wrapping (one line per cell) ----------
                                # --- Styling: dark table + no wrapping (one line per cell) ----------
                                BG_DARK    = "#1e2630"
                                BG_DARKER  = "#171d26"
                                TEXT_COLOR = "#e6eaf0"
                                GRID_COLOR = "rgba(255,255,255,0.08)"
                                ROW_DIVIDER = "#D1D5DB"  # light grey bottom border between rows

                                table_styles = [
                                    {"selector": "table", "props": [
                                        ("background-color", BG_DARK),
                                        ("color", TEXT_COLOR),
                                        ("border-collapse", "separate"),
                                        ("border-spacing", "0"),
                                        ("font-size", "0.95rem"),
                                        ("table-layout", "auto"),
                                    ]},
                                    {"selector": "thead th", "props": [
                                        ("background-color", BG_DARKER),
                                        ("color", TEXT_COLOR),
                                        ("border-bottom", f"1px solid {GRID_COLOR}"),
                                        ("border-top", f"1px solid {GRID_COLOR}"),
                                        ("padding", "10px 12px"),
                                        ("font-weight", "600"),
                                        ("letter-spacing", "0.02em"),
                                        ("white-space", "nowrap"),
                                        ("text-align", "center"),        # <-- center header text
                                    ]},
                                    {"selector": "tbody td, tbody th.row_heading", "props": [
                                        ("border-bottom", f"1px solid {ROW_DIVIDER}"),  # <-- was GRID_COLOR
                                        ("padding", "10px 12px"),
                                        ("vertical-align", "middle"),
                                        ("white-space", "nowrap"),
                                        ("max-width", "none"),
                                        ("text-align", "center"),
                                    ]},
                                ]

                                # === Filled chip styles for sentiment labels (both columns) =========
                                # === Sentiment text colors only (no background/border) ==============
                                SENT_GREEN = "#1F7A3E"   # Positive
                                SENT_RED   = "#B3392F"   # Negative
                                SENT_AMBER = "#8A6D3B"   # Neutral
                                


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
                                            css.append("")  # default styling
                                    return css





                                styled = (
                                    table.style
                                        .set_table_styles(table_styles)
                                        .hide(axis="index")  # keep if you’re hiding the DF index
                                        .set_properties(
                                            subset=["Row", "Date", "Image sentiment", "Text sentiment", "Risk score", "Post ID"],
                                            **{
                                                "white-space": "nowrap",
                                                "text-align": "center",
                                                "font-family": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                                            }
                                        )
                                        .set_properties(subset=["Post ID"], **{"font-family": "ui-monospace, Menlo, Consolas, monospace"})
                                        .apply(style_sentiment_color, subset=["Image sentiment", "Text sentiment"])
                                )




                                # --- Cap card height with vertical scroll (keeps your styling) ------
                                CARD_MAX_HEIGHT = 420  # match height of your other cards

                                import streamlit.components.v1 as components

                                html = f"""
                                <style>
                                html, body {{
                                    background: transparent;
                                    margin: 0; padding: 0;
                                    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; /* prevent Times fallback */
                                }}
                                .wrapper {{
                                    max-height: {CARD_MAX_HEIGHT}px;   /* cap height like other cards */
                                    overflow-y: auto;                  /* vertical scroll within card */
                                    overflow-x: auto;                  /* horizontal if needed */
                                    border: 1px solid rgba(255,255,255,0.08);
                                    border-radius: 10px;
                                    background: transparent;
                                }}
                                table {{ width: 100%; border-collapse: separate; border-spacing: 0; }}
                                thead th {{ position: sticky; top: 0; z-index: 1; background: {BG_DARKER}; font-family: inherit; }}
                                tbody td, tbody th {{ font-family: inherit; }}
                                </style>
                                <div class="wrapper">
                                {styled.to_html()}
                                </div>
                                """

                                # Disable iframe scrolling; let inner div handle it
                                components.html(html, height=CARD_MAX_HEIGHT + 24, scrolling=False)

        except Exception as e:
            st.warning(f"Heatmap unavailable: {e}")









