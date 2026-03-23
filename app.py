import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Orbital SIGINT Console", layout="wide")

st.title("Orbital SIGINT Console")
st.caption(
    "Aerospace activity dashboard for launches, sensitive missions, failures, and satellite watchlist monitoring."
)

# =========================================================
# CONFIG
# =========================================================
WATCHLIST = [
    {"watch_name": "ISS (ZARYA)", "NORAD_CAT_ID": "25544", "role": "Crewed station"},
    {"watch_name": "CSS (TIANHE)", "NORAD_CAT_ID": "48274", "role": "Crewed station"},
    {"watch_name": "NOAA 15", "NORAD_CAT_ID": "25338", "role": "Weather"},
    {"watch_name": "HRC MONOBLOCK CAMERA", "NORAD_CAT_ID": "62386", "role": "Imaging / unknown"},
]

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


def get_secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def clean_time_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not df.empty and col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


# =========================================================
# LAUNCH DATA
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
# SPACE-TRACK
# =========================================================
@st.cache_data(ttl=1800)
def get_spacetrack_watchlist_status(watchlist_records):
    identity = get_secret("spacetrack_identity")
    password = get_secret("spacetrack_password")

    watch_df = pd.DataFrame(watchlist_records).copy()
    watch_df["NORAD_CAT_ID"] = watch_df["NORAD_CAT_ID"].astype(str)

    if not identity or not password:
        watch_df["owner"] = "Not loaded"
        watch_df["object_type"] = "Not loaded"
        watch_df["launch_year"] = pd.NA
        watch_df["country_code"] = pd.NA
        watch_df["decay_epoch"] = pd.NaT
        watch_df["health"] = "Secrets missing"
        watch_df["severity"] = "MEDIUM"
        watch_df["analyst_note"] = "Add Space-Track secrets to enable catalog and decay checks."
        return watch_df

    session = requests.Session()

    login_url = "https://www.space-track.org/ajaxauth/login"
    login_payload = {"identity": identity, "password": password}
    login_resp = session.post(login_url, data=login_payload, timeout=30)
    login_resp.raise_for_status()

    ids = ",".join(watch_df["NORAD_CAT_ID"].tolist())

    satcat_url = (
        "https://www.space-track.org/basicspacedata/query/"
        f"class/satcat/NORAD_CAT_ID/{ids}/orderby/NORAD_CAT_ID asc/format/json"
    )

    decay_url = (
        "https://www.space-track.org/basicspacedata/query/"
        f"class/decay/NORAD_CAT_ID/{ids}/orderby/DECAY_EPOCH desc/format/json"
    )

    satcat_resp = session.get(satcat_url, timeout=30)
    satcat_resp.raise_for_status()
    satcat_df = pd.DataFrame(satcat_resp.json())

    decay_resp = session.get(decay_url, timeout=30)
    decay_resp.raise_for_status()
    decay_df = pd.DataFrame(decay_resp.json())

    if satcat_df.empty:
        satcat_df = pd.DataFrame(columns=["NORAD_CAT_ID"])
    if decay_df.empty:
        decay_df = pd.DataFrame(columns=["NORAD_CAT_ID", "DECAY_EPOCH"])

    satcat_df["NORAD_CAT_ID"] = satcat_df["NORAD_CAT_ID"].astype(str)
    decay_df["NORAD_CAT_ID"] = decay_df["NORAD_CAT_ID"].astype(str)

    if "LAUNCH" in satcat_df.columns:
        satcat_df["LAUNCH"] = pd.to_datetime(satcat_df["LAUNCH"], errors="coerce")
    if "DECAY" in satcat_df.columns:
        satcat_df["DECAY"] = pd.to_datetime(satcat_df["DECAY"], errors="coerce")
    if "DECAY_EPOCH" in decay_df.columns:
        decay_df["DECAY_EPOCH"] = pd.to_datetime(decay_df["DECAY_EPOCH"], utc=True, errors="coerce")

    satcat_keep = [
        c
        for c in [
            "NORAD_CAT_ID",
            "OBJECT_NAME",
            "OBJECT_TYPE",
            "COUNTRY",
            "COUNTRY_CODE",
            "LAUNCH",
            "SITE",
            "DECAY",
        ]
        if c in satcat_df.columns
    ]
    satcat_df = satcat_df[satcat_keep].drop_duplicates("NORAD_CAT_ID")

    decay_keep = [c for c in ["NORAD_CAT_ID", "DECAY_EPOCH"] if c in decay_df.columns]
    decay_df = decay_df[decay_keep].sort_values("DECAY_EPOCH", ascending=False).drop_duplicates("NORAD_CAT_ID")

    merged = watch_df.merge(satcat_df, how="left", on="NORAD_CAT_ID")
    merged = merged.merge(decay_df, how="left", on="NORAD_CAT_ID")

    merged["owner"] = merged.get("COUNTRY")
    merged["object_type"] = merged.get("OBJECT_TYPE")
    merged["country_code"] = merged.get("COUNTRY_CODE")
    merged["launch_year"] = merged["LAUNCH"].dt.year if "LAUNCH" in merged.columns else pd.NA
    merged["decay_epoch"] = merged.get("DECAY_EPOCH")

    def classify_health(row):
        has_cat = pd.notna(row.get("OBJECT_NAME"))
        has_decay = pd.notna(row.get("DECAY_EPOCH")) or pd.notna(row.get("DECAY"))
        if has_decay:
            return "Offline / decayed"
        if has_cat:
            return "Cataloged / no decay record"
        return "Unknown"

    def severity_from_health(health):
        if health == "Offline / decayed":
            return "HIGH"
        if health == "Unknown":
            return "MEDIUM"
        if health == "Secrets missing":
            return "MEDIUM"
        return "LOW"

    def analyst_note(row):
        health = row["health"]
        if health == "Offline / decayed":
            return "Decay or reentry record present in public data."
        if health == "Cataloged / no decay record":
            return "Object is present in public catalog and has no decay record."
        if health == "Secrets missing":
            return "Space-Track credentials not configured."
        return "No public catalog match found for this NORAD ID."

    merged["health"] = merged.apply(classify_health, axis=1)
    merged["severity"] = merged["health"].apply(severity_from_health)
    merged["analyst_note"] = merged.apply(analyst_note, axis=1)

    final_cols = [
        "watch_name",
        "NORAD_CAT_ID",
        "role",
        "owner",
        "country_code",
        "object_type",
        "launch_year",
        "health",
        "severity",
        "decay_epoch",
        "analyst_note",
    ]
    return merged[final_cols]


# =========================================================
# LOAD DATA
# =========================================================
launch_error = None
recent_launch_error = None
satellite_error = None

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

try:
    satellites_df = get_spacetrack_watchlist_status(WATCHLIST)
except Exception as e:
    satellites_df = pd.DataFrame(WATCHLIST)
    satellite_error = str(e)

# =========================================================
# DERIVED TABLES
# =========================================================
failed_launches_df = pd.DataFrame()
sensitive_launches_df = pd.DataFrame()
offline_satellites_df = pd.DataFrame()

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

if not satellites_df.empty and "health" in satellites_df.columns:
    offline_satellites_df = satellites_df[
        satellites_df["health"].isin(["Offline / decayed", "Unknown"])
    ].copy()

# =========================================================
# STATUS CARDS
# =========================================================
st.subheader("System Status")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("Upcoming Launches", len(launches_df))
with c2:
    st.metric("Failed Launches", len(failed_launches_df))
with c3:
    st.metric("Sensitive Launches", len(sensitive_launches_df))
with c4:
    st.metric("Watched Satellites", len(satellites_df))
with c5:
    st.metric("Offline / Unknown Sats", len(offline_satellites_df))
with c6:
    if launch_error:
        st.error("Launch Feed Offline")
    else:
        st.success("Launch Feed Online")

st.divider()

# =========================================================
# MAP + SATELLITE PANEL
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
    st.subheader("Satellite Layer")
    st.caption("Health below is inferred from public catalog + decay/reentry data, not onboard telemetry.")

    if satellite_error:
        st.error(f"Satellite watchlist error: {satellite_error}")
    elif satellites_df.empty:
        st.warning("No satellite watchlist data loaded.")
    else:
        low_count = int((satellites_df["severity"] == "LOW").sum())
        med_count = int((satellites_df["severity"] == "MEDIUM").sum())
        high_count = int((satellites_df["severity"] == "HIGH").sum())

        s1, s2, s3 = st.columns(3)
        s1.metric("Low", low_count)
        s2.metric("Medium", med_count)
        s3.metric("High", high_count)

        owner_counts = (
            satellites_df["owner"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )
        owner_counts.columns = ["owner", "count"]

        if not owner_counts.empty:
            fig_owner = px.bar(
                owner_counts,
                x="owner",
                y="count",
                title="Watchlist by Owner",
            )
            fig_owner.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_owner, use_container_width=True)

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
# SATELLITE WATCHLIST TABLE
# =========================================================
st.subheader("Satellite Watchlist Health & Ownership")

if satellite_error:
    st.error(f"Satellite watchlist unavailable: {satellite_error}")
elif satellites_df.empty:
    st.info("No watched satellite data available.")
else:
    sat_display = satellites_df.copy()
    sat_display = sat_display.rename(
        columns={
            "watch_name": "Satellite",
            "NORAD_CAT_ID": "NORAD",
            "role": "Role",
            "owner": "Owner",
            "country_code": "Country",
            "object_type": "Object Type",
            "launch_year": "Launch Year",
            "health": "Health",
            "severity": "Severity",
            "decay_epoch": "Decay / Reentry",
            "analyst_note": "Analyst Note",
        }
    )
    st.dataframe(sat_display, use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# OFFLINE / UNKNOWN SATELLITES
# =========================================================
st.subheader("Satellites Flagged Offline or Unknown")

if satellite_error:
    st.error(f"Satellite status unavailable: {satellite_error}")
elif offline_satellites_df.empty:
    st.success("No watched satellites are currently flagged offline or unknown.")
else:
    off_display = offline_satellites_df.copy().rename(
        columns={
            "watch_name": "Satellite",
            "NORAD_CAT_ID": "NORAD",
            "role": "Role",
            "owner": "Owner",
            "country_code": "Country",
            "object_type": "Object Type",
            "launch_year": "Launch Year",
            "health": "Health",
            "severity": "Severity",
            "decay_epoch": "Decay / Reentry",
            "analyst_note": "Analyst Note",
        }
    )
    st.dataframe(off_display, use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# ANALYST SUMMARY
# =========================================================
st.subheader("Analyst Summary")

next_launch_name = safe_text(launches_df.iloc[0]["name"]) if not launches_df.empty else "No launch available"
next_launch_time = safe_text(launches_df.iloc[0]["net"]) if not launches_df.empty else "N/A"

if satellites_df.empty:
    sat_summary = "Satellite watchlist data is not currently available."
else:
    sat_summary = (
        f"{len(satellites_df)} watched satellites checked, "
        f"{len(offline_satellites_df)} flagged offline or unknown."
    )

st.markdown(
    f"""
- **{len(launches_df)}** upcoming launch records are currently loaded.
- **{len(failed_launches_df)}** failed launches were found in the last 30 days.
- **{len(sensitive_launches_df)}** publicly labeled sensitive launches were found in the last 90 days.
- The **next scheduled launch** is **{next_launch_name}**.
- The next launch time is **{next_launch_time}**.
- **Satellite watchlist:** {sat_summary}
- Satellite “health” here is a **public-data inference**, not real-time spacecraft telemetry.
"""
)
