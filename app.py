import os
import json
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


st.set_page_config(page_title="Orbital Launch Monitor", layout="wide")

st.title("Orbital Launch Monitor")
st.caption(
    "A live dashboard for upcoming launches, recent failed launches, publicly labeled sensitive missions, and AI-assisted OSINT inference."
)


# CONFIG

SENSITIVE_KEYWORDS = [
    "government/top secret",
    "top secret",
    "government",
    "national security",
    "military",
    "reconnaissance",
    "surveillance",
    "classified",
    "nrol",
    "usa-",
    "yaogan",
]

WATCHED_PROVIDERS = [
    "united launch alliance",
    "northrop grumman",
    "spacex",
    "rocket lab",
    "roscosmos",
]


# HELPERS

def safe_text(value):
    return "" if value is None else str(value)


def clean_time_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not df.empty and col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def get_openai_api_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def classify_sensitive_launch(row: pd.Series) -> dict:
    """
    Rule-based fallback inference using only public metadata.
    This version aims to always provide a mission-specific explanation,
    not a generic 'classified/national security' answer.
    """
    name = safe_text(row.get("name")).lower()
    mission_type = safe_text(row.get("mission_type")).lower()
    provider = safe_text(row.get("provider")).lower()
    rocket = safe_text(row.get("rocket")).lower()
    location_name = safe_text(row.get("location_name")).lower()
    country_code = safe_text(row.get("country_code")).lower()
    status = safe_text(row.get("status")).lower()

    text = " ".join([name, mission_type, provider, rocket, location_name, country_code, status])

    evidence = []
    likely_type = "Sensitive government payload"
    why_sensitive = (
        "This mission most likely supports government or military space operations, "
        "and missions in that category are often described vaguely to avoid revealing capability, customer, or purpose."
    )
    confidence = 0.55

    # Mission-family-specific explanations
    if "nrol" in text:
        likely_type = "Reconnaissance / intelligence satellite"
        why_sensitive = (
            "NROL missions are typically linked to the U.S. National Reconnaissance Office, "
            "which operates intelligence-gathering satellites. Launches in this family are usually kept vague "
            "because they can involve imaging, signals collection, or other surveillance capabilities."
        )
        confidence = 0.86
        evidence.append("Mission name includes NROL")

    elif "yaogan" in text:
        likely_type = "Remote sensing / military ISR satellite"
        why_sensitive = (
            "Yaogan missions are widely associated with Chinese reconnaissance and remote sensing activity. "
            "They are often treated as sensitive because they may support surveillance, maritime tracking, "
            "or broader military intelligence functions."
        )
        confidence = 0.82
        evidence.append("Mission name includes Yaogan")

    elif "reconnaissance" in text:
        likely_type = "Reconnaissance / surveillance payload"
        why_sensitive = (
            "The mission is publicly described as reconnaissance, which strongly suggests an intelligence, "
            "monitoring, or ISR role. Payloads in this category are often sensitive because they reveal how a state "
            "collects information from orbit."
        )
        confidence = 0.82
        evidence.append("Mission type or name includes reconnaissance")

    elif "surveillance" in text:
        likely_type = "Surveillance / ISR payload"
        why_sensitive = (
            "The mission is labeled as surveillance, which usually points to monitoring, tracking, "
            "or intelligence-gathering activity. Missions like this are often kept partially vague because "
            "their collection targets or technical capability may be strategically important."
        )
        confidence = 0.79
        evidence.append("Mission type or name includes surveillance")

    elif "national security" in text:
        likely_type = "National security space mission"
        why_sensitive = (
            "The mission is publicly framed as a national security launch, which usually means it supports "
            "defense, intelligence, missile warning, secure communications, or orbital monitoring. "
            "Those missions are often sensitive because detailed disclosure could expose strategic capability."
        )
        confidence = 0.78
        evidence.append("Public metadata includes national security wording")

    elif "classified" in text or "top secret" in text or "secret" in text:
        likely_type = "Classified government mission"
        why_sensitive = (
            "The mission is explicitly described as classified or secret, which usually indicates a payload tied to "
            "defense, intelligence, or another restricted government function. Public details are often limited to avoid "
            "revealing the system’s exact role or capability."
        )
        confidence = 0.77
        evidence.append("Public metadata includes classified/secret wording")

    elif "military" in text:
        likely_type = "Military support payload"
        why_sensitive = (
            "The mission appears linked to military activity, which often means it supports communications, surveillance, "
            "navigation, early warning, or other operational functions. Payloads in these roles are commonly described vaguely "
            "to avoid exposing their usefulness in conflict or strategic planning."
        )
        confidence = 0.73
        evidence.append("Public metadata includes military wording")

    elif "government" in text:
        likely_type = "Government operational payload"
        why_sensitive = (
            "The mission appears to be a government payload rather than a commercial one. Government missions are often more "
            "sensitive because they can support state communications, observation, science with dual-use value, or other public-sector operations."
        )
        confidence = 0.68
        evidence.append("Public metadata includes government wording")

    # Provider / launcher pattern refinements
    if "spacex" in text and "nrol" in text:
        evidence.append("Pattern resembles recent Falcon 9 NRO mission profiles")
        confidence = max(confidence, 0.86)

    if "united launch alliance" in text or "ula" in text:
        evidence.append("ULA frequently launches U.S. national security payloads")
        if likely_type == "Sensitive government payload":
            likely_type = "Possible U.S. national security payload"
            why_sensitive = (
                "This mission uses a launch provider strongly associated with U.S. government and defense launches. "
                "That makes it more likely to involve a payload with military, intelligence, or strategic support value."
            )
        confidence = max(confidence, 0.73)

    if "northrop grumman" in text:
        evidence.append("Northrop Grumman commonly supports defense missions")
        if likely_type == "Sensitive government payload":
            likely_type = "Possible defense-linked payload"
            why_sensitive = (
                "This mission is associated with a contractor that regularly supports defense and national security programs, "
                "which increases the chance that the payload has military or intelligence relevance."
            )
        confidence = max(confidence, 0.72)

    if "rocket lab" in text and ("government" in text or "military" in text):
        evidence.append("Rocket Lab increasingly launches security-linked payloads")
        confidence = max(confidence, 0.66)

    if "roscosmos" in text and ("military" in text or "government" in text):
        evidence.append("Roscosmos missions can include state-directed or dual-use payloads")
        confidence = max(confidence, 0.66)

    if "vandenberg" in text:
        evidence.append("Vandenberg launches often support polar-orbit security missions")

    if "cape canaveral" in text or "kennedy" in text:
        evidence.append("Florida launch sites are often used for strategic U.S. government missions")

    if not evidence:
        evidence.append("Inference based on public mission labels and launch pattern")

    return {
        "likely_type": likely_type,
        "why_sensitive": why_sensitive,
        "confidence": round(confidence, 2),
        "evidence": evidence[:4],
    }


def infer_sensitive_launch_ai(row: pd.Series, model: str = "gpt-5.2") -> dict:
    """
    AI explanation layer. Falls back to rules if API key or package is unavailable.
    """
    api_key = get_openai_api_key()
    if not OPENAI_AVAILABLE or not api_key:
        return classify_sensitive_launch(row)

    client = OpenAI(api_key=api_key)

    fields = {
        "name": safe_text(row.get("name")),
        "net": safe_text(row.get("net")),
        "status": safe_text(row.get("status")),
        "provider": safe_text(row.get("provider")),
        "rocket": safe_text(row.get("rocket")),
        "mission_type": safe_text(row.get("mission_type")),
        "location_name": safe_text(row.get("location_name")),
        "country_code": safe_text(row.get("country_code")),
    }

    prompt = f"""
You are an aerospace OSINT analyst.

A launch has been flagged as publicly sensitive based on public metadata.
Using ONLY the metadata below, infer the most likely mission category and explain specifically why a mission of this type would be considered sensitive.

Rules:
- Do not claim certainty.
- Use cautious language.
- Do not give a generic answer like "it is classified" or "it is national security-related."
- Explain the likely operational reason it is sensitive, such as reconnaissance, ISR, military communications, missile warning, government monitoring, or strategic support.
- Base your answer only on the metadata below.
- Return valid JSON only.

Required JSON schema:
{{
  "likely_type": "string",
  "why_sensitive": "string",
  "confidence": 0.0,
  "evidence": ["string", "string", "string"]
}}

Launch metadata:
{json.dumps(fields, ensure_ascii=False)}
"""

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        parsed = json.loads(response.output_text.strip())

        return {
            "likely_type": parsed.get("likely_type", "Unknown sensitive payload"),
            "why_sensitive": parsed.get("why_sensitive", "Public metadata suggests a security-linked mission."),
            "confidence": float(parsed.get("confidence", 0.5)),
            "evidence": parsed.get("evidence", [])[:4],
        }
    except Exception:
        return classify_sensitive_launch(row)


# DATA LOADING

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


# LOAD DATA

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


# DERIVED TABLES

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
        | (
            provider_series.apply(lambda x: any(k in x for k in WATCHED_PROVIDERS))
            & (
                mission_series.str.contains(
                    "government|military|surveillance|reconnaissance|classified|national security",
                    na=False,
                )
                | name_series.str.contains(
                    "nrol|usa-|classified|secret|military|yaogan",
                    na=False,
                )
            )
        )
    )

    sensitive_launches_df = recent_launches_df[sensitive_mask].copy()
    sensitive_launches_df = sensitive_launches_df[
        sensitive_launches_df["net"] >= now_utc - pd.Timedelta(days=90)
    ].copy()


# STATUS CARDS

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


# MAP + OVERVIEW PANEL

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
    st.write("- AI-assisted mission inference")

st.divider()


# UPCOMING LAUNCHES

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


# RECENT FAILED LAUNCHES

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


# SENSITIVE LAUNCHES

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


# AI INFERENCE ON SENSITIVE LAUNCHES

st.subheader("AI Inference on Sensitive Launches")
st.caption(
    "Best-effort OSINT estimate of why a publicly labeled sensitive mission may be sensitive. This is not confirmation of the real payload."
)

if recent_launch_error:
    st.error(f"Recent launch history unavailable: {recent_launch_error}")
elif sensitive_launches_df.empty:
    st.info("No sensitive launches available for AI inference.")
else:
    use_ai = st.toggle("Use AI-generated explanations", value=True)
    max_rows = st.slider(
        "Number of sensitive launches to analyse",
        min_value=1,
        max_value=min(10, len(sensitive_launches_df)),
        value=min(5, len(sensitive_launches_df)),
    )

    analysis_df = sensitive_launches_df.head(max_rows).copy()
    results = []

    with st.spinner("Analysing sensitive missions..."):
        for _, row in analysis_df.iterrows():
            result = infer_sensitive_launch_ai(row) if use_ai else classify_sensitive_launch(row)

            results.append(
                {
                    "Launch": safe_text(row.get("name")),
                    "Time (UTC)": safe_text(row.get("net")),
                    "Provider": safe_text(row.get("provider")),
                    "Rocket": safe_text(row.get("rocket")),
                    "Mission Type": safe_text(row.get("mission_type")),
                    "Likely Purpose": result["likely_type"],
                    "Why It May Be Sensitive": result["why_sensitive"],
                    "Confidence": f"{int(float(result['confidence']) * 100)}%",
                    "Evidence": " | ".join(result["evidence"]),
                }
            )

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    with st.expander("Show detailed analyst cards"):
        for _, row in results_df.iterrows():
            st.markdown(f"### {row['Launch']}")
            st.write(f"**Time (UTC):** {row['Time (UTC)']}")
            st.write(f"**Provider:** {row['Provider']}")
            st.write(f"**Rocket:** {row['Rocket']}")
            st.write(f"**Mission Type:** {row['Mission Type']}")
            st.write(f"**Likely Purpose:** {row['Likely Purpose']}")
            st.write(f"**Why It May Be Sensitive:** {row['Why It May Be Sensitive']}")
            st.write(f"**Confidence:** {row['Confidence']}")
            st.write(f"**Evidence:** {row['Evidence']}")
            st.markdown("---")

st.divider()


# ANALYST SUMMARY

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
- Sensitive-mission inference is enabled with rule-based logic and optional AI explanation support.
""")
