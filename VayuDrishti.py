"""
VayuDrishti — AWS Anomaly Detection Dashboard (SIH26073 prototype)

Run it with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_FILE = "final_frontend_output.csv"

st.set_page_config(page_title="VayuDrishti", layout="wide")

# ---------------------------------------------------------------------------
# 1. STATION SETUP
# ---------------------------------------------------------------------------
STATIONS = {
    "BHOPAL": {"name": "Bhopal", "region": "Central", "lat": 23.2599, "lon": 77.4126},
    "DEWAS": {"name": "Dewas", "region": "West", "lat": 22.9676, "lon": 76.0534},
    "DHAR": {"name": "Dhar", "region": "West", "lat": 22.5979, "lon": 75.2975},
    "INDORE": {"name": "Indore", "region": "West", "lat": 22.7196, "lon": 75.8577},
    "UJJAIN": {"name": "Ujjain", "region": "West", "lat": 23.1765, "lon": 75.7885},
}
HOURS = 30


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# 2. GENERATE / LOAD DATA
# ---------------------------------------------------------------------------
def load_station_data_from_csv(path):
    """Load trained VayuDrishti backend output without deleting dropout rows."""
    raw = pd.read_csv(path, parse_dates=["timestamp"])
    required = {
        "station", "timestamp", "temperature", "humidity", "pressure",
        "status", "final_anomaly", "final_fault_type", "severity",
        "confidence", "sensor_health", "temporal_score", "spatial_score",
    }
    missing = required - set(raw.columns)
    if missing:
        st.error(f"Your CSV is missing required column(s): {sorted(missing)}")
        st.stop()

    raw["station_id"] = raw["station"].astype(str).str.upper()
    raw["temp"] = pd.to_numeric(raw["temperature"], errors="coerce")
    raw["humidity"] = pd.to_numeric(raw["humidity"], errors="coerce")
    raw["pressure"] = pd.to_numeric(raw["pressure"], errors="coerce")
    raw["final_anomaly"] = raw["final_anomaly"].astype(str).str.lower().eq("true")
    raw["final_fault_type"] = raw["final_fault_type"].fillna("").astype(str).str.lower()
    raw["status"] = raw["status"].fillna("Healthy").astype(str)
    raw["severity"] = raw["severity"].fillna("Low").astype(str)
    raw["confidence"] = pd.to_numeric(raw["confidence"], errors="coerce").fillna(0)
    raw["sensor_health"] = pd.to_numeric(raw["sensor_health"], errors="coerce").fillna(100)
    raw["temporal_score"] = pd.to_numeric(raw["temporal_score"], errors="coerce").fillna(0).astype(int)
    raw["spatial_score"] = pd.to_numeric(raw["spatial_score"], errors="coerce").fillna(0).astype(int)

    data = {}
    for sid in STATIONS:
        sdf = raw[raw["station_id"] == sid].sort_values("timestamp").reset_index(drop=True).copy()
        if sdf.empty:
            continue
        sdf["hour"] = range(len(sdf))
        sdf["is_frozen"] = sdf["final_fault_type"].eq("frozen")
        sdf["is_anomaly"] = sdf["final_anomaly"]
        data[sid] = sdf

    if not data:
        st.error("No valid stations found matching the VayuDrishti station IDs.")
        st.stop()
    return data

if "data" not in st.session_state:
    if os.path.exists(DATA_FILE):
        st.session_state.data = load_station_data_from_csv(DATA_FILE)
    else:
        st.session_state.data = generate_station_data(seed=42)

ACTIVE_STATIONS = {sid: meta for sid, meta in STATIONS.items() if sid in st.session_state.data}
if "time_idx" not in st.session_state:
    st.session_state.time_idx = 10
if "playing" not in st.session_state:
    st.session_state.playing = False
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_station" not in st.session_state:
    st.session_state.selected_station = "BHOPAL"

max_idx = min(len(df) for df in st.session_state.data.values()) - 1
st.session_state.time_idx = min(st.session_state.time_idx, max_idx)


# ---------------------------------------------------------------------------
# 3. BACKEND STATUS / FAULT LABELS
# ---------------------------------------------------------------------------
FAULT_LABELS = {
    "": "Healthy",
    "normal": "Healthy",
    "frozen": "Faulty: frozen sensor",
    "drift": "Drift watch",
    "dropout": "Faulty: dropout",
    "spike": "Faulty: spike",
    "noise_burst": "Faulty: noise burst",
}

FAULT_FILTERS = [
    "Healthy",
    "Drift watch",
    "Faulty: frozen sensor",
    "Faulty: dropout",
    "Faulty: spike",
    "Faulty: noise burst",
]

def row_diagnostics(df, idx):
    row = df.loc[idx]
    fault = str(row.get("final_fault_type", "")).lower()
    label = FAULT_LABELS.get(fault, "Healthy" if not bool(row.get("final_anomaly", False)) else fault.title())
    return {
        "status": label,
        "fault_type": "Healthy" if not fault else fault.replace("_", " ").title(),
        "severity": str(row.get("severity", "Low")),
        "confidence": float(row.get("confidence", 0)),
        "sensor_health": float(row.get("sensor_health", 100)),
        "temporal_score": int(row.get("temporal_score", 0)),
        "spatial_score": int(row.get("spatial_score", 0)),
        "anomaly": bool(row.get("final_anomaly", False)),
    }

def compute_status(df, idx):
    d = row_diagnostics(df, idx)
    return d["status"], d["confidence"]

# Kept as a UI testing helper; it does not alter the trained backend output.
def inject_fault(station_id, idx):
    st.session_state.data[station_id].loc[idx:, "is_frozen"] = True


# 4. HOME PAGE
# ---------------------------------------------------------------------------
def render_home():
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>VayuDrishti</h1>"
        "<p style='text-align:center; color:gray;'>AI/ML anomaly detection for Automatic Weather Stations</p>",
        unsafe_allow_html=True,
    )

    @st.fragment(run_every=0.6 if st.session_state.playing else None)
    def live_metrics():
        if st.session_state.playing:
            if st.session_state.time_idx < max_idx:
                st.session_state.time_idx += 1
            else:
                st.session_state.playing = False
        idx = st.session_state.time_idx
        n_faulty = 0
        reliabilities = []
        for sid in ACTIVE_STATIONS:
            df = st.session_state.data[sid]
            df.name = sid
            status, reliability = compute_status(df, idx)
            reliabilities.append(reliability)
            if status != "Healthy":
                n_faulty += 1
        c1, c2, c3 = st.columns(3)
        c1.metric("Stations online", f"{len(ACTIVE_STATIONS)}/{len(ACTIVE_STATIONS)}")
        c2.metric("Active faults / watches", n_faulty)
        c3.metric("Avg reliability", f"{np.mean(reliabilities):.0f}%")

    live_metrics()

    st.caption("🔍 Search or click on any station row below to view its detailed inspection.")
    search = st.text_input(
        "Search station id or location",
        "",
        key="search_box",
        label_visibility="collapsed",
        placeholder="Search by station name or ID",
    )

    # --- DATE PICKER AND CONTROL BUTTONS ---
    sample_df = st.session_state.data[list(ACTIVE_STATIONS.keys())[0]]
    
    if "timestamp" in sample_df.columns:
        min_date = sample_df["timestamp"].min().date()
        max_date = sample_df["timestamp"].max().date()
        current_date = sample_df.loc[st.session_state.time_idx, "timestamp"].date()
    else:
        import datetime
        min_date, max_date, current_date = datetime.date.today(), datetime.date.today(), datetime.date.today()

    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        chosen_date = st.date_input("Jump to date", value=current_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
        if chosen_date != current_date and "timestamp" in sample_df.columns:
            match_idx = sample_df[sample_df["timestamp"].dt.date == chosen_date].index
            if len(match_idx) > 0:
                st.session_state.time_idx = int(match_idx[0])
                st.rerun()
    with bcol2:
        label = "⏸ Pause" if st.session_state.playing else "▶ Play"
        if st.button(label, use_container_width=True):
            st.session_state.playing = not st.session_state.playing
            st.rerun()
    with bcol3:
        if st.button("⏭ Next network fault", use_container_width=True):
            found = False
            idx = st.session_state.time_idx
            for sid in ACTIVE_STATIONS:
                df = st.session_state.data[sid]
                df.name = sid
                for j in range(idx + 1, len(df)):
                    status, _ = compute_status(df, j)
                    if status != "Healthy":
                        st.session_state.time_idx = j
                        found = True
                        break
                if found:
                    break
            if not found:
                st.toast("No further faults ahead in the timeline.")
            st.rerun()
    with bcol4:
        with st.popover("🔻 Filters", use_container_width=True):
            st.multiselect("Fault Type / Status", FAULT_FILTERS, key="status_filter")
            st.multiselect("Region", ["Central", "West"], key="region_filter")

    @st.fragment(run_every=0.6 if st.session_state.playing else None)
    def live_dashboard():
        idx = st.session_state.time_idx
        rows = []
        map_data = []
        row_station_ids = []

        for sid, meta in ACTIVE_STATIONS.items():
            df = st.session_state.data[sid]
            d = row_diagnostics(df, idx)
            row = df.loc[idx]
            time_display = row["timestamp"].strftime("%Y-%m-%d %H:00")
            rows.append({
                "Station": f"{sid}, {meta['name']}",
                "Region": meta.get("region", "Unknown"),
                "Timestamp": time_display,
                "Temperature (°C)": "—" if pd.isna(row["temp"]) else round(row["temp"], 1),
                "Humidity (%)": "—" if pd.isna(row["humidity"]) else round(row["humidity"], 1),
                "Pressure (hPa)": "—" if pd.isna(row["pressure"]) else round(row["pressure"], 1),
                "Status": d["status"],
                "Fault Type": d["fault_type"],
                "Severity": d["severity"] if d["anomaly"] else "—",
                "Confidence (%)": round(d["confidence"], 1) if d["anomaly"] else 0.0,
                "Temporal Score": d["temporal_score"],
                "Spatial Score": d["spatial_score"],
                "Sensor Health (%)": round(d["sensor_health"], 1),
            })
            row_station_ids.append(sid)
            severity = d["severity"].lower()
            if not d["anomaly"]:
                marker_color = "#2fae6b"
            elif severity in ("critical", "very high"):
                marker_color = "#8b0000"
            elif severity == "high":
                marker_color = "#e34948"
            elif severity == "medium":
                marker_color = "#e0a300"
            else:
                marker_color = "#7a5af8"
            map_data.append({
                "lat": meta["lat"], "lon": meta["lon"], "color": marker_color,
                "hover_text": f"<b>{sid} - {meta['name']}</b><br>Status: {d['status']}<br>Fault: {d['fault_type']}<br>Confidence: {d['confidence']:.1f}%"
            })

        table = pd.DataFrame(rows)
        search_val = st.session_state.get("search_box", "")
        status_filter = st.session_state.get("status_filter", [])
        region_filter = st.session_state.get("region_filter", [])
        keep = pd.Series(True, index=table.index)
        if search_val:
            keep &= table["Station"].str.contains(search_val, case=False, na=False)
        if status_filter:
            keep &= table["Status"].isin(status_filter)
        if region_filter:
            keep &= table["Region"].isin(region_filter)
        table = table.loc[keep].reset_index(drop=True)
        visible_ids = [sid for sid, k in zip(row_station_ids, keep.tolist()) if k]

        def highlight_faults(row):
            status = row["Status"]
            if status == "Healthy":
                return [""] * len(row)
            if status == "Drift watch":
                bg = "background-color: rgba(224, 163, 0, 0.25)"
            elif status == "Faulty: dropout":
                bg = "background-color: rgba(122, 90, 248, 0.22)"
            elif status == "Faulty: noise burst":
                bg = "background-color: rgba(42, 120, 214, 0.20)"
            else:
                bg = "background-color: rgba(227, 73, 72, 0.25)"
            return [bg] * len(row)

        styled_table = table.style.apply(highlight_faults, axis=1)
        st.markdown(
            f"<span style='background:#5b4ee8; color:white; padding:4px 12px; border-radius:12px; font-size:13px;'>LIVE — Backend replay (hour {idx})</span>",
            unsafe_allow_html=True,
        )
        event = st.dataframe(
            styled_table, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row", key="station_table",
        )
        st.caption("Tap/click any station row to open its detailed inspection.")
        picked_rows = event.selection.rows if event and event.selection else []
        if picked_rows and picked_rows[0] < len(visible_ids):
            st.session_state.selected_station = visible_ids[picked_rows[0]]
            st.session_state.page = "detail"
            st.rerun()

        st.markdown("### Live Station Map")
        map_df = pd.DataFrame(map_data)
        fig = go.Figure(go.Scattermap(
            lat=map_df["lat"], lon=map_df["lon"], mode="markers",
            marker=dict(size=14, color=map_df["color"], opacity=0.9),
            text=map_df["hover_text"], hoverinfo="text"
        ))
        fig.update_layout(
            map_style="carto-positron", hovermode="closest",
            map=dict(center=dict(lat=map_df["lat"].mean(), lon=map_df["lon"].mean()), zoom=5.5),
            margin={"r":0,"t":10,"l":0,"b":0}, height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    live_dashboard()

# 5. STATION DETAIL PAGE
# ---------------------------------------------------------------------------
def render_detail():
    sid = st.session_state.selected_station
    meta = STATIONS[sid]
    df = st.session_state.data[sid]
    idx = st.session_state.time_idx
    d = row_diagnostics(df, idx)

    if st.button("← Back to network"):
        st.session_state.page = "home"
        st.rerun()

    status = d["status"]
    color_map = {
        "Healthy": "#2fae6b", "Drift watch": "#e0a300",
        "Faulty: frozen sensor": "#e34948", "Faulty: dropout": "#7a5af8",
        "Faulty: spike": "#e34948", "Faulty: noise burst": "#2a78d6",
    }
    color = color_map.get(status, "#e34948")

    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown(f"## {sid}, {meta['name']}")
        st.caption(f"Region: {meta.get('region', 'Unknown')}  •  Backend fault classification: {d['fault_type']}")
    with hcol2:
        st.markdown(f"<div style='text-align:right;'><span style='background:{color}22; color:{color}; padding:4px 12px; border-radius:8px; font-size:13px;'>{status}</span></div>", unsafe_allow_html=True)

    faults_so_far = int(df.loc[:idx, "final_anomaly"].sum())
    uptime_pct = 100.0 if idx == 0 else 100.0 * (1 - faults_so_far / (idx + 1))
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Current Status", status)
    scol2.metric("Detected anomalies", faults_so_far)
    scol3.metric("Uptime (%)", f"{uptime_pct:.1f}%")

    st.divider()
    st.markdown("### Sensor Telemetry")
    # Show telemetry for the selected calendar day only, not the entire 2020–2025 history.
    selected_day = df.loc[idx, "timestamp"].date()
    shown = df[df["timestamp"].dt.date == selected_day].copy()
    point_colors = [color_map.get(FAULT_LABELS.get(str(f).lower(), "Healthy"), "#e34948") if a else "#2a78d6"
                    for f, a in zip(shown["final_fault_type"], shown["final_anomaly"])]
    tab1, tab2, tab3 = st.tabs(["Temperature", "Humidity", "Pressure"])

    def create_telemetry_fig(y_data, y_title, series_name):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=shown["timestamp"], y=y_data, mode="lines+markers",
            line=dict(color="#2a78d6", width=2), marker=dict(color=point_colors, size=8), name=series_name
        ))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Time", yaxis_title=y_title)
        return fig

    with tab1:
        st.plotly_chart(create_telemetry_fig(shown["temp"], "°C", "Temperature"), use_container_width=True)
    with tab2:
        st.plotly_chart(create_telemetry_fig(shown["humidity"], "%", "Humidity"), use_container_width=True)
    with tab3:
        st.plotly_chart(create_telemetry_fig(shown["pressure"], "hPa", "Pressure"), use_container_width=True)
    st.caption("🔴/🟣/🔵 markers indicate backend-detected anomalous readings; blue line shows telemetry.")

    st.markdown("### Backend Diagnostics")
    diag = pd.DataFrame([{
        "Fault Type": d["fault_type"], "Severity": d["severity"] if d["anomaly"] else "—",
        "Confidence (%)": round(d["confidence"], 1), "Temporal Score": d["temporal_score"],
        "Spatial Score": d["spatial_score"], "Sensor Health (%)": round(d["sensor_health"], 1),
    }])
    st.table(diag)

    st.markdown("### Event log")
    if d["anomaly"]:
        row = df.loc[idx]
        st.table(pd.DataFrame([{
            "Station": sid, "Timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "Fault Type": d["fault_type"], "Severity": d["severity"],
            "Confidence (%)": round(d["confidence"], 1), "Temporal Score": d["temporal_score"],
            "Spatial Score": d["spatial_score"], "Sensor Health (%)": round(d["sensor_health"], 1),
        }]))
    else:
        st.success("No active backend-detected fault for this station at the selected time.")

    st.markdown(f"### Neighbor comparison at {df.loc[idx, 'timestamp'].strftime('%Y-%m-%d %H:%M')}")
    dists = []
    for other_id, other_meta in ACTIVE_STATIONS.items():
        if other_id == sid:
            continue
        dist = haversine_km(meta["lat"], meta["lon"], other_meta["lat"], other_meta["lon"])
        dists.append((other_id, other_meta["name"], dist))
    dists.sort(key=lambda x: x[2])
    rows = []
    for oid, oname, dist in [(sid, meta["name"], 0)] + dists[:2]:
        odf = st.session_state.data[oid]
        r = odf.loc[idx]
        rows.append({
            "Station": f"{oid} ({oname})", "Distance": "—" if oid == sid else f"{dist:.0f} km",
            "Temp": "—" if pd.isna(r["temp"]) else f"{r['temp']:.1f}°C",
            "Humidity": "—" if pd.isna(r["humidity"]) else f"{r['humidity']:.1f}%",
            "Pressure": "—" if pd.isna(r["pressure"]) else f"{r['pressure']:.1f} hPa",
            "Status": row_diagnostics(odf, idx)["status"],
        })
    st.table(pd.DataFrame(rows))

    st.markdown("### Station Controls")
    if st.button("⏭ Jump to next flagged entry", use_container_width=True):
        found = False
        for j in range(idx + 1, len(df)):
            if bool(df.loc[j, "final_anomaly"]):
                st.session_state.time_idx = j
                found = True
                break
        if not found:
            st.toast("No further backend-detected faults ahead for this station.")
        st.rerun()

    st.markdown("### Why the physical consistency check matters")
    st.caption("A genuine regional weather event should often appear consistently across nearby stations, while an isolated sensor fault may not.")

# 6. ROUTER
# ---------------------------------------------------------------------------
if st.session_state.page == "home":
    render_home()
else:
    render_detail()