import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Orbital Launch Monitor", layout="wide")

st.title("Orbital Launch Monitor")
st.caption("A live dashboard for upcoming launches, recent failed launches, and publicly labeled sensitive missions.")

# =========================================================
# CONFIG
# =========================================================
SENSITIVE_KEYWORDS = [
    "government/top secret",
    "top secret",
    "government",
    "national security",
    "military",
    "reconnaissance",
    "surveillance",
    "classified",
]

WATCHED_PROVIDERS = [
    "united launch alliance",
    "northrop grumman",
    "spacex",
    "rocket lab",
    "roscosmos",
]

# =========================================================
# HELPERS
# =========================================================
def safe_text(value):
    return "" if value is None else str(value)

def clean_time_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not df.empty and col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(ttl=300)
def get_upcoming_launches():
    url = "https://ll.thespacedevs.com/2.2.0/launch/upcoming/?limit=15&mode=detailed"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    raw = response.json()["results"]

    rows = []
    for item in raw:
        pad = item.get("pad") or {}
        location = pad.get("location") or {}

        rows.append(
            {
                "name": item.get("name"),
                "net": item.get("net"),
                "status": item.get("status", {}).get("name") if item.get("status") else None,
                "provider": item.get("launch_service_provider", {}).get("name")
                if item.get("launch_service_provider")
                else None,
                "rocket": item.get("rocket", {}).get("configuration", {}).get("name")
                if item.get("rocket") and item.get("rocket", {}).get("configuration")
                else None,
                "mission_type": item.get("mission", {}).get("type") if item.get("mission") else None,
                "location_name": location.get("name"),
                "country_code": location.get("country_code"),
                "lat": pd.to_numeric(pad.get("latitude"), errors="coerce"),
                "lon": pd.to_numeric(pad.get("longitude"), errors="coerce"),
            }
        )

    df = pd.DataFrame(rows)
    df = clean_time_col(df, "net")
    if not df.empty:
        df = df.sort_values("net")
    return df


@st.cache_data(ttl=300)
def get_recent_launches():
    url = "https://ll.thespacedevs.com/2.2.0/launch/previous/?limit=60&mode=detailed"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    raw = response.json()["results"]

    rows = []
    for item in raw:
        pad = item.get("pad") or {}
        location = pad.get("location") or {}

        rows.append(
            {
                "name": item.get("name"),
                "net": item.get("net"),
                "status": item.get("status", {}).get("name") if item.get("status") else None,
                "provider": item.get("launch_service_provider", {}).get("name")
                if item.get("launch_service_provider")
                else None,
                "rocket": item.get("rocket", {}).get("configuration", {}).get("name")
                if item.get("rocket") and item.get("rocket", {}).get("configuration")
                else None,
                "mission_type": item.get("mission", {}).get("type") if item.get("mission") else None,
                "location_name": location.get("name"),
                "country_code": location.get("country_code"),
            }
        )

    df = pd.DataFrame(rows)
    df = clean_time_col(df, "net")
    if not df.empty:
        df = df.sort_values("net", ascending=False)
    return df

# =========================================================
# LOAD DATA
# =========================================================
launch_error = None
recent_launch_error = None

try:
    launches_df = get_upcoming_launches()
except Exception as e:
    launches_df = pd.DataFrame()
    launch_error = str(e)

try:
    recent_launches_df = get_recent_launches()
except Exception as e:
    recent_launches_df = pd.DataFrame()
    recent_launch_error = str(e)

# =========================================================
# DERIVED TABLES
# =========================================================
failed_launches_df = pd.DataFrame()
sensitive_launches_df = pd.DataFrame()

if not recent_launches_df.empty:
    now_utc = pd.Timestamp.utcnow()

    failed_keywords = ["failure", "partial failure", "failed"]
    failed_mask = (
        recent_launches_df["status"]
        .fillna("")
        .str.lower()
        .apply(lambda x: any(word in x for word in failed_keywords))
    )
    failed_launches_df = recent_launches_df[failed_mask].copy()
    failed_launches_df = failed_launches_df[
        failed_launches_df["net"] >= now_utc - pd.Timedelta(days=30)
    ].copy()

    mission_series = recent_launches_df["mission_type"].fillna("").str.lower()
    provider_series = recent_launches_df["provider"].fillna("").str.lower()
    name_series = recent_launches_df["name"].fillna("").str.lower()

    sensitive_mask = (
        mission_series.apply(lambda x: any(k in x for k in SENSITIVE_KEYWORDS))
        | name_series.apply(lambda x: any(k in x for k in SENSITIVE_KEYWORDS))
        | provider_series.apply(lambda x: any(k in x for k in WATCHED_PROVIDERS))
    )

    sensitive_launches_df = recent_launches_df[sensitive_mask].copy()
    sensitive_launches_df = sensitive_launches_df[
        sensitive_launches_df["net"] >= now_utc - pd.Timedelta(days=90)
    ].copy()

# =========================================================
# STATUS CARDS
# =========================================================
st.subheader("System Status")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Upcoming Launches", len(launches_df))

with c2:
    st.metric("Recent Failed Launches", len(failed_launches_df))

with c3:
    st.metric("Sensitive Launches", len(sensitive_launches_df))

with c4:
    if launch_error:
        st.error("Launch Feed Offline")
    else:
        st.success("Launch Feed Online")

st.divider()

# =========================================================
# MAP + OVERVIEW PANEL
# =========================================================
left, right = st.columns([1.5, 1])

with left:
    st.subheader("Launch Site Map")

    if launch_error:
        st.error(f"Launch map unavailable: {launch_error}")
    elif launches_df.empty:
        st.info("No launch data available right now.")
    else:
        map_df = launches_df.dropna(subset=["lat", "lon"]).copy()
        if map_df.empty:
            st.info("No launch coordinates available right now.")
        else:
            fig_map = px.scatter_geo(
                map_df,
                lat="lat",
                lon="lon",
                hover_name="name",
                hover_data=["provider", "rocket", "mission_type", "location_name", "net"],
                title="Upcoming Launch Locations",
            )
            fig_map.update_traces(marker=dict(size=10))
            fig_map.update_layout(height=520, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_map, use_container_width=True)

with right:
    st.subheader("Launch Overview")

    if launch_error:
        st.error("Live launch feed is currently unavailable.")
    else:
        st.success("Live launch monitoring is active.")

        if not launches_df.empty:
            next_launch_name = safe_text(launches_df.iloc[0]["name"])
            next_launch_time = safe_text(launches_df.iloc[0]["net"])

            st.markdown("**Next scheduled launch**")
            st.write(next_launch_name)
            st.write(next_launch_time)

    st.markdown("**Current focus**")
    st.write("- Upcoming launch activity")
    st.write("- Recent failed launches")
    st.write("- Publicly labeled sensitive missions")

st.divider()

# =========================================================
# UPCOMING LAUNCHES
# =========================================================
st.subheader("Upcoming Launches")

if launch_error:
    st.error(f"Launch feed unavailable: {launch_error}")
elif launches_df.empty:
    st.info("No upcoming launches available.")
else:
    nice_launches = launches_df[
        ["name", "net", "status", "provider", "rocket", "mission_type", "location_name"]
    ].copy()
    nice_launches = nice_launches.rename(
        columns={
            "name": "Launch",
            "net": "Time (UTC)",
            "status": "Status",
            "provider": "Provider",
            "rocket": "Rocket",
            "mission_type": "Mission Type",
            "location_name": "Location",
        }
    )
    st.dataframe(nice_launches, use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# RECENT FAILED LAUNCHES
# =========================================================
st.subheader("Recent Failed Launches")

if recent_launch_error:
    st.error(f"Recent launch history unavailable: {recent_launch_error}")
elif failed_launches_df.empty:
    st.success("No failed launches found in the last 30 days.")
else:
    failed_display = failed_launches_df[
        ["name", "net", "status", "provider", "rocket", "mission_type", "location_name"]
    ].copy()
    failed_display = failed_display.rename(
        columns={
            "name": "Launch",
            "net": "Time (UTC)",
            "status": "Status",
            "provider": "Provider",
            "rocket": "Rocket",
            "mission_type": "Mission Type",
            "location_name": "Location",
        }
    )
    st.dataframe(failed_display, use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# SENSITIVE LAUNCHES
# =========================================================
st.subheader("Publicly Labeled Sensitive Launches")
st.caption("This section uses public labels and metadata only. It does not identify undisclosed or covert launches.")

if recent_launch_error:
    st.error(f"Recent launch history unavailable: {recent_launch_error}")
elif sensitive_launches_df.empty:
    st.info("No publicly labeled sensitive launches found in the last 90 days.")
else:
    sensitive_display = sensitive_launches_df[
        ["name", "net", "status", "provider", "rocket", "mission_type", "location_name"]
    ].copy()
    sensitive_display = sensitive_display.rename(
        columns={
            "name": "Launch",
            "net": "Time (UTC)",
            "status": "Status",
            "provider": "Provider",
            "rocket": "Rocket",
            "mission_type": "Mission Type",
            "location_name": "Location",
        }
    )
    st.dataframe(sensitive_display, use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# ANALYST SUMMARY
# =========================================================
st.subheader("Analyst Summary")

if launch_error:
    st.markdown("""
- Launch feed is currently unavailable.
- The dashboard layout is online, but live data sources need attention.
""")
else:
    next_launch_name = safe_text(launches_df.iloc[0]["name"]) if not launches_df.empty else "No launch available"
    next_launch_time = safe_text(launches_df.iloc[0]["net"]) if not launches_df.empty else "N/A"

    st.markdown(f"""
- **{len(launches_df)}** upcoming launch records are currently loaded.
- **{len(failed_launches_df)}** failed launches were found in the last 30 days.
- **{len(sensitive_launches_df)}** publicly labeled sensitive launches were found in the last 90 days.
- The **next scheduled launch** is **{next_launch_name}**.
- The next launch time is **{next_launch_time}**.
- The launch layer is operational.
""")

