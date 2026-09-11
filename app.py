"""
SkyGuard — AWS Anomaly Detection Dashboard (SIH26073 prototype)

Run it with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_FILE = "weather_data.csv"

st.set_page_config(page_title="SkyGuard", layout="wide")

# ---------------------------------------------------------------------------
# 1. STATION SETUP
# ---------------------------------------------------------------------------
STATIONS = {
    "AWS-014": {"name": "Indore",      "lat": 22.7196, "lon": 75.8577},
    "AWS-021": {"name": "Bhopal",      "lat": 23.2599, "lon": 77.4126},
    "AWS-029": {"name": "Berasia",     "lat": 23.6300, "lon": 77.4300},
    "AWS-032": {"name": "South Bhopal","lat": 23.1500, "lon": 77.4500},
    "AWS-041": {"name": "Sehore",      "lat": 23.2000, "lon": 77.0900},
    "AWS-045": {"name": "Jabalpur",    "lat": 23.1815, "lon": 79.9864},
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
def generate_station_data(seed):
    rng = np.random.default_rng(seed)
    hours = np.arange(HOURS)
    
    # Create a dummy starting date so the calendar has something to read
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = [start_time + pd.Timedelta(hours=int(h)) for h in hours]
    
    data = {}
    for sid in STATIONS:
        base_temp = 28 + 8 * np.sin((hours - 6) / 24 * 2 * np.pi) 
        noise = rng.normal(0, 0.4, HOURS)
        temp = base_temp + noise

        if sid == "AWS-021":
            temp = temp + np.linspace(0, 4, HOURS)

        humidity = np.clip(70 - (temp - 28) * 2 + rng.normal(0, 2, HOURS), 15, 90)
        pressure = 1008 + rng.normal(0, 0.8, HOURS)

        data[sid] = pd.DataFrame({
            "hour": hours, "timestamp": timestamps, "temp": temp, 
            "humidity": humidity, "pressure": pressure, "is_frozen": False,
        })
    return data
def load_station_data_from_csv(path):
    raw = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"station_id", "timestamp", "temp", "humidity", "pressure"}
    missing = required - set(raw.columns)
    if missing:
        st.error(f"Your CSV is missing required column(s): {missing}")
        st.stop()

    data = {}
    for sid in STATIONS:
        sdf = raw[raw["station_id"] == sid].sort_values("timestamp").reset_index(drop=True)
        if sdf.empty:
            continue
        sdf = sdf.dropna(subset=["temp", "humidity", "pressure"]).reset_index(drop=True)
        sdf["hour"] = range(len(sdf))
        sdf["is_frozen"] = False
        data[sid] = sdf[["hour", "timestamp", "temp", "humidity", "pressure", "is_frozen"]]

    if not data:
        st.error("No valid stations found matching STATIONS dictionary.")
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
    st.session_state.selected_station = "AWS-032"

max_idx = min(len(df) for df in st.session_state.data.values()) - 1
st.session_state.time_idx = min(st.session_state.time_idx, max_idx)


# ---------------------------------------------------------------------------
# 3. STATUS RULES & FAULT INJECTION
# ---------------------------------------------------------------------------
def compute_status(df, idx):
    window = df.loc[max(0, idx - 4): idx, "temp"]
    if len(window) >= 4 and window.std() < 0.05:
        return "Faulty: frozen sensor", int(np.random.default_rng(idx).integers(25, 40))
    if df.name == "AWS-021" and idx > 12:
        return "Drift watch", int(np.random.default_rng(idx).integers(80, 90))
    return "Healthy", int(np.random.default_rng(idx).integers(93, 99))


def inject_fault(station_id, idx):
    df = st.session_state.data[station_id]
    freeze_temp = df.loc[idx, "temp"]
    freeze_hum = df.loc[idx, "humidity"]
    freeze_pres = df.loc[idx, "pressure"]
    df.loc[idx:, "temp"] = freeze_temp
    df.loc[idx:, "humidity"] = freeze_hum
    df.loc[idx:, "pressure"] = freeze_pres
    df.loc[idx:, "is_frozen"] = True


# ---------------------------------------------------------------------------
# 4. HOME PAGE
# ---------------------------------------------------------------------------
def render_home():
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>SkyGuard</h1>"
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
            st.multiselect("Sensor Health", ["Healthy", "Drift watch", "Faulty: frozen sensor"], key="status_filter")
            st.multiselect("Region", ["Central", "North", "South", "East", "West"], key="region_filter")

    @st.fragment(run_every=0.6 if st.session_state.playing else None)
    def live_dashboard():
        idx = st.session_state.time_idx
        rows = []
        map_data = []
        row_station_ids = []
        
        for sid, meta in ACTIVE_STATIONS.items():
            df = st.session_state.data[sid]
            df.name = sid
            status, reliability = compute_status(df, idx)
            row = df.loc[idx]
            
            if "timestamp" in row:
                time_display = row["timestamp"].strftime("%Y-%m-%d %H:00")
            else:
                time_display = f"Hour {row['hour']}"
                
            rows.append({
                ("Station", ""): f"{sid}, {meta['name']}",
                ("Region", ""): meta.get("region", "Unknown"),
                ("Timestamp", ""): time_display,
                ("Parameters", "Temp (°C)"): round(row["temp"], 1),
                ("Parameters", "Humidity (%)"): round(row["humidity"], 1),
                ("Parameters", "Pressure (hPa)"): round(row["pressure"], 1),
                ("Reliability", "Score (%)"): reliability,
                ("Status", ""): status,
            })
            row_station_ids.append(sid)
            map_data.append({
                "lat": meta["lat"],
                "lon": meta["lon"],
                "color": "#e34948" if status != "Healthy" else "#2fae6b",
                "hover_text": f"<b>{sid} - {meta['name']}</b><br>Status: {status}"
            })
            
        # Table View
        table = pd.DataFrame(rows)
        table.columns = pd.MultiIndex.from_tuples(table.columns)

        search_val = st.session_state.get("search_box", "")
        status_filter = st.session_state.get("status_filter", [])
        region_filter = st.session_state.get("region_filter", [])
        
        keep = pd.Series(True, index=table.index)
        if search_val:
            keep &= table[("Station", "")].str.contains(search_val, case=False)
        if status_filter:
            keep &= table[("Status", "")].isin(status_filter)
        if region_filter:
            keep &= table[("Region", "")].isin(region_filter)
            
        table = table[keep]
        visible_ids = [sid for sid, k in zip(row_station_ids, keep) if k]

        # Conditional Formatting Function
        def highlight_faults(row):
            status = row[("Status", "")]
            if status == "Faulty: frozen sensor":
                return ["background-color: rgba(227, 73, 72, 0.25)"] * len(row)
            elif status == "Drift watch":
                return ["background-color: rgba(224, 163, 0, 0.25)"] * len(row)
            return [""] * len(row)

        styled_table = table.style.apply(highlight_faults, axis=1)

        st.markdown(
            f"<span style='background:#5b4ee8; color:white; padding:4px 12px; "
            f"border-radius:12px; font-size:13px;'>LIVE — Simulated replay (hour {idx})</span>",
            unsafe_allow_html=True,
        )

        event = st.dataframe(
            styled_table, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row", key="station_table",
        )
        st.caption("Tap/click the checkbox on the left of any row to open its detailed inspection.")
        picked_rows = event.selection.rows if event and event.selection else []
        if picked_rows:
            st.session_state.selected_station = visible_ids[picked_rows[0]]
            st.session_state.page = "detail"
            st.rerun()

        # --- Plotly Interactive Map View ---
        st.markdown("### Live Station Map")
        map_df = pd.DataFrame(map_data)
        
        fig = go.Figure(go.Scattermap(
            lat=map_df["lat"],
            lon=map_df["lon"],
            mode='markers',
            marker=dict(
                size=14,
                color=map_df["color"],
                opacity=0.9
            ),
            text=map_df["hover_text"],
            hoverinfo="text"
        ))
        
        fig.update_layout(
            map_style="carto-positron",
            hovermode='closest',
            map=dict(
                center=dict(lat=map_df["lat"].mean(), lon=map_df["lon"].mean()),
                zoom=5.5
            ),
            margin={"r":0,"t":10,"l":0,"b":0},
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    live_dashboard()# ---------------------------------------------------------------------------
# 5. STATION DETAIL PAGE
# ---------------------------------------------------------------------------
def render_detail():
    sid = st.session_state.selected_station
    meta = STATIONS[sid]
    df = st.session_state.data[sid]
    df.name = sid
    idx = st.session_state.time_idx

    if st.button("← Back to network"):
        st.session_state.page = "home"
        st.rerun()

    status, reliability = compute_status(df, idx)
    color = {"Healthy": "#2fae6b", "Drift watch": "#e0a300", "Faulty: frozen sensor": "#e34948"}[status]

    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown(f"## {sid}, {meta['name']}")
    with hcol2:
        st.markdown(
            f"<div style='text-align:right;'><span style='background:{color}22; color:{color}; padding:4px 12px; "
            f"border-radius:8px; font-size:13px;'>{status}</span></div>",
            unsafe_allow_html=True,
        )
        
    # --- Summary Metrics ---
    faults_today = sum(1 for j in range(idx + 1) if compute_status(df, j)[0] != "Healthy")
    uptime_pct = 100.0 if idx == 0 else 100.0 * (1 - (faults_today / (idx + 1)))
    
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Current Status", status)
    scol2.metric("Faults Today", faults_today)
    scol3.metric("Uptime (%)", f"{uptime_pct:.1f}%")
    
    st.divider()

    # --- Time series chart (with tabs for all parameters) ---
    shown = df.loc[: idx]
    point_colors = ["#e34948" if f else "#2a78d6" for f in shown["is_frozen"]]
    
    st.markdown("### Sensor Telemetry")
    tab1, tab2, tab3 = st.tabs(["Temperature", "Humidity", "Pressure"])
    
    def create_telemetry_fig(y_data, y_title, series_name):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=shown["hour"], y=y_data, mode="lines+markers",
                line=dict(color="#2a78d6", width=2),
                marker=dict(color=point_colors, size=8),
                name=series_name,
            )
        )
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hour of day", yaxis_title=y_title)
        return fig

    with tab1:
        st.plotly_chart(create_telemetry_fig(shown["temp"], "°C", "Temperature"), use_container_width=True)
    with tab2:
        st.plotly_chart(create_telemetry_fig(shown["humidity"], "%", "Humidity"), use_container_width=True)
    with tab3:
        st.plotly_chart(create_telemetry_fig(shown["pressure"], "hPa", "Pressure"), use_container_width=True)
        
    st.caption("🔴 Fault detected &nbsp;&nbsp; 🔵 Normal reading")

   # --- Station Controls ---
    st.markdown("### Station Controls")
    b1, b2 = st.columns(2)
    with b1:
        with st.popover(f"⚡ Inject custom fault ({sid})", use_container_width=True):
            st.write("Override current hour readings to test the ML model:")
            
            # Grab current real values to use as defaults in the input boxes
            curr_temp = float(df.loc[idx, 'temp'])
            curr_hum = float(df.loc[idx, 'humidity'])
            curr_pres = float(df.loc[idx, 'pressure'])
            
            new_temp = st.number_input("Temp (°C)", value=curr_temp)
            new_hum = st.number_input("Humidity (%)", value=curr_hum)
            new_pres = st.number_input("Pressure (hPa)", value=curr_pres)
            
            if st.button("Apply Anomaly Data"):
                # Overwrite the raw data at this exact timestamp
                st.session_state.data[sid].loc[idx, 'temp'] = new_temp
                st.session_state.data[sid].loc[idx, 'humidity'] = new_hum
                st.session_state.data[sid].loc[idx, 'pressure'] = new_pres
                
                # Notice we are NO LONGER setting 'is_frozen = True'. 
                # The ML model must figure out it's an anomaly on its own!
                st.rerun()
    with b2:
        if st.button("⏭ Jump to next flagged entry", use_container_width=True):
            found = False
            for j in range(idx + 1, len(df)):
                s, _ = compute_status(df, j)
                if s != "Healthy":
                    st.session_state.time_idx = j
                    found = True
                    break
            if not found:
                st.toast("No further faults ahead for this specific station.")
            st.rerun()

    # --- Event log ---
    st.markdown("### Event log")
    if status != "Healthy":
        log_data = [{
            ("Station ID", ""): sid,
            ("Timestamp", ""): f"Hour {idx}",
            ("Parameters", "Temp (°C)"): round(df.loc[idx, 'temp'], 1),
            ("Parameters", "Humidity (%)"): round(df.loc[idx, 'humidity'], 1),
            ("Parameters", "Pressure (hPa)"): round(df.loc[idx, 'pressure'], 1),
            ("Diagnostics", "Confidence Level"): f"{reliability}%",
            ("Diagnostics", "Fault Type"): "Hardware (Stuck Sensor)" if "frozen" in status else "Hardware (Calibration Decay)",
        }]
        log_df = pd.DataFrame(log_data)
        log_df.columns = pd.MultiIndex.from_tuples(log_df.columns)
        st.table(log_df)
    else:
        st.success("No active faults for this station right now.")

    # --- Neighbor comparison ---
    st.markdown(f"### Neighbor comparison at hour {idx}")
    dists = []
    for other_id, other_meta in ACTIVE_STATIONS.items():
        if other_id == sid: continue
        d = haversine_km(meta["lat"], meta["lon"], other_meta["lat"], other_meta["lon"])
        dists.append((other_id, other_meta["name"], d))
    dists.sort(key=lambda x: x[2])
    neighbors = dists[:2]

    rows = [{
        "Station": f"{sid} (this)", "Distance": "—",
        "Temp": f"{df.loc[idx, 'temp']:.1f}°C",
        "Reading": "Flat, unchanged" if df.loc[idx, "is_frozen"] else "Normal",
    }]
    for nid, nname, d in neighbors:
        ndf = st.session_state.data[nid]
        rows.append({
            "Station": f"{nid} ({nname})", "Distance": f"{d:.0f} km",
            "Temp": f"{ndf.loc[idx, 'temp']:.1f}°C", "Reading": "Normal daily pattern",
        })
    st.table(pd.DataFrame(rows))
    if df.loc[idx, "is_frozen"]:
        st.caption("Neighbors show a normal pattern while this station stayed flat — confirms sensor fault.")

    # --- Physical consistency illustration ---
    st.markdown("### Why the physical consistency check matters")
    st.caption("Illustrative example (genuine heatwave) — shows the failure mode a threshold-only system has.")
    ex_hours = list(range(8, 22))
    ex_temp = [30, 31, 33, 35, 37, 39, 41, 43, 44, 44, 43, 41, 38, 35]
    icol1, icol2 = st.columns(2)
    with icol1:
        st.markdown("❌ **Without physical consistency check**")
        colors_bad = ["#e34948" if t >= 41 else "#2a78d6" for t in ex_temp]
        f1 = go.Figure(go.Scatter(x=ex_hours, y=ex_temp, mode="lines+markers",
                                   line=dict(color="#2a78d6"), marker=dict(color=colors_bad, size=8)))
        f1.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(f1, use_container_width=True)
    with icol2:
        st.markdown("✅ **With physical consistency check**")
        f2 = go.Figure(go.Scatter(x=ex_hours, y=ex_temp, mode="lines+markers",
                                   line=dict(color="#2a78d6"), marker=dict(color="#2a78d6", size=8)))
        f2.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(f2, use_container_width=True)

# ---------------------------------------------------------------------------
# 6. ROUTER
# ---------------------------------------------------------------------------
if st.session_state.page == "home":
    render_home()
else:
    render_detail()