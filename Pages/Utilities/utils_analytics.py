import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Constants & Styling
# =============================================================================
DEFAULT_STATIC = os.path.join(os.path.dirname(__file__), "Static")
STATIC_DIR = st.session_state.get("storage_paths", {}).get("STATIC_DIR", DEFAULT_STATIC)

# Color palette
COLOR_POS = "#21A366"
COLOR_NEU = "#F1C40F"
COLOR_NEG = "#E74C3C"
COLOR_MAP = {"Positive": COLOR_POS, "Neutral": COLOR_NEU, "Negative": COLOR_NEG}

# Table theme colors
BG_DARK = "#1e2630"
BG_DARKER = "#171d26"
TEXT_COLOR = "#e6eaf0"
GRID_COLOR = "rgba(255,255,255,0.08)"
ROW_DIVIDER = "#D1D5DB"

TABLE_STYLES = [
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
        ("text-align", "center"),
    ]},
    {"selector": "tbody td, tbody th.row_heading", "props": [
        ("border-bottom", f"1px solid {ROW_DIVIDER}"),
        ("padding", "10px 12px"),
        ("vertical-align", "middle"),
        ("white-space", "nowrap"),
        ("max-width", "none"),
        ("text-align", "center"),
    ]},
]


# =============================================================================
# File & Profile Helpers
# =============================================================================
def list_available_users(static_dir: str):
    """Return a list of usernames that have both profile.json and metadata.json."""
    if not os.path.isdir(static_dir):
        return []
    users = []
    for name in os.listdir(static_dir):
        user_dir = os.path.join(static_dir, name)
        if not os.path.isdir(user_dir):
            continue
        has_profile = os.path.exists(os.path.join(user_dir, "profile.json"))
        has_meta = os.path.exists(os.path.join(user_dir, "metadata.json"))
        if has_profile and has_meta:
            users.append(name)
    return sorted(users, key=str.lower)


def load_profile_bundle(username: str):
    """Load user profile data and return dict + image path if found."""
    user_dir = os.path.join(STATIC_DIR, username)
    profile_json_path = os.path.join(user_dir, "profile.json")
    profile_pic_path = os.path.join(user_dir, "profile.jpg")
    if not os.path.exists(profile_json_path):
        raise FileNotFoundError(f"No profile.json for {username} in {user_dir}")
    with open(profile_json_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    if not os.path.exists(profile_pic_path):
        profile_pic_path = None
    return profile, profile_pic_path


def load_meta_records(username: str):
    """Load post metadata for given username."""
    meta_path = os.path.join(STATIC_DIR, username, "metadata.json")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_profile_header(profile: dict, profile_pic_path: str | None):
    """Render centered header with profile picture and stats."""
    _left, mid, _right = st.columns([1, 2, 1])
    with mid:
        with st.container(border=False):
            st.markdown('<div style="padding: 12px 16px;">', unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")


# =============================================================================
# Calendar Heatmap Helpers
# =============================================================================
def build_calendar_matrix_from_meta(meta_records, year: int) -> pd.DataFrame:
    """Convert metadata list to 7×N matrix of daily post counts for heatmap."""
    if not meta_records:
        return pd.DataFrame()

    df = pd.DataFrame(meta_records)
    if df.empty or "taken_at" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["taken_at"], errors="coerce", utc=True).dt.date
    df = df[pd.to_datetime(df["date"]).dt.year == year]

    daily = df.groupby("date").size().rename("count").reset_index()
    full_year = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    cal = pd.DataFrame({"date": full_year})
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)
    cal = cal.merge(daily, on="date", how="left").fillna({"count": 0})
    cal["count"] = cal["count"].astype(int)
    cal["dow"] = cal["date"].dt.weekday
    cal["week_start"] = (cal["date"] - pd.to_timedelta(cal["dow"], unit="D")).dt.date
    pivot = cal.pivot_table(index="dow", columns="week_start", values="count", fill_value=0)
    return pivot.reindex(index=range(0, 7))


def calendar_heatmap_figure(pivot: pd.DataFrame, cell_px=22, gaps=(1, 1)) -> go.Figure:
    """Return Plotly heatmap with adaptive x-range and hover info."""
    if pivot.empty:
        return go.Figure()

    x = [pd.to_datetime(str(d)) for d in pivot.columns]
    y_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    z = pivot.values

    # Range padding logic
    nonzero = pivot.columns[(pivot > 0).any(axis=0)]
    d_start = pd.to_datetime(str(pivot.columns.min()))
    d_end = pd.to_datetime(str(pivot.columns.max())) + pd.Timedelta(weeks=1)
    if len(nonzero):
        dmin, dmax = pd.to_datetime(str(nonzero.min())), pd.to_datetime(str(nonzero.max()))
        pad = pd.Timedelta(days=7)
        six = pd.Timedelta(days=183)
        span = dmax - dmin
        mid = dmin + span / 2
        half = max(six / 2, span / 2 + pad)
        start, end = max(mid - half, d_start), min(mid + half, d_end)
    else:
        mid = d_start + (d_end - d_start) / 2
        half = pd.Timedelta(days=183) / 2
        start, end = max(d_start, mid - half), min(d_end, mid + half)

    hover_text = [
        [f"{(pd.to_datetime(str(w)) + pd.Timedelta(days=r)).date()}: {z[r][c]} post(s)"
         for c, w in enumerate(pivot.columns)]
        for r in range(7)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z, x=x, y=y_labels,
            text=hover_text, hoverinfo="text",
            coloraxis="coloraxis",
            xgap=gaps[0], ygap=gaps[1],
        )
    )
    fig.update_layout(
        margin=dict(l=50, r=70, t=40, b=40),
        height=pivot.shape[0] * cell_px + 80,
        coloraxis=dict(colorscale="BuGn", colorbar=dict(x=1.02)),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    fig.update_xaxes(type="date", range=[start, end], dtick="M1", tickformat="%b", showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# =============================================================================
# DataFrame Utility Helpers
# =============================================================================
def pick_timestamp_col(df: pd.DataFrame) -> str | None:
    """Return timestamp column if found."""
    if "taken_at" in df.columns:
        return "taken_at"
    if "date_utc" in df.columns:
        return "date_utc"
    return None


def to_year_df(meta_records: list, year: int) -> pd.DataFrame:
    """Return dataframe filtered to a given calendar year."""
    df = pd.DataFrame(meta_records)
    ts = pick_timestamp_col(df)
    if not ts:
        return pd.DataFrame()
    df[ts] = pd.to_datetime(df[ts], errors="coerce", utc=True)
    return df[df[ts].dt.year == int(year)].copy()


def label_from_prediction_cell(x) -> str:
    """Extract image label from dict, string, or list."""
    if x is None:
        return ""
    if isinstance(x, dict):
        return str(x.get("label", "")).strip()
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        for i in x:
            if isinstance(i, dict) and i.get("label"):
                return str(i["label"]).strip()
            if isinstance(i, str) and i.strip():
                return i.strip()
    return str(x).strip()


def sentiment_series(df: pd.DataFrame, label_col="sentiment_label", score_col="sentiment_score") -> pd.Series:
    """Return unified sentiment labels (Positive, Neutral, Negative)."""
    if label_col in df.columns and df[label_col].notna().any():
        raw = df[label_col].astype(str).str.lower().str.strip()
    elif score_col in df.columns:
        s = df[score_col]
        raw = pd.Series(pd.NA, index=df.index)
        raw = raw.mask(s >= 0.6, "positive").mask(s <= 0.4, "negative").fillna("neutral")
    else:
        return pd.Series(index=df.index, dtype="object")

    return raw.map({"positive": "Positive", "neutral": "Neutral", "negative": "Negative"})


def metric_int(x) -> int:
    """Safely cast to int."""
    try:
        return int(x)
    except Exception:
        return 0
