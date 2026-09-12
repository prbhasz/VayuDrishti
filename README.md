# VayuDrishti
SkyGuard: An AI/ML-powered anomaly detection dashboard for Automatic Weather Station (AWS) networks. Built for SIH26073.
# SkyGuard 🌤️
**AI/ML Anomaly Detection for Automatic Weather Stations (AWS)**
*Developed for Smart India Hackathon (SIH26073)*

## Overview
SkyGuard is a real-time monitoring and anomaly detection dashboard designed to manage large-scale telemetry data from weather station networks. It processes high-volume environmental data (Temperature, Humidity, Pressure) and visualizes machine-learning predictions to instantly flag hardware faults, sensor drifts, and frozen data points.

## Key Features
* **Live Replay Engine:** Simulates real-time data ingestion for historical datasets, allowing operators to scrub through timelines.
* **Interactive Mapping:** Geographic tracking of network health using Plotly interactive maps.
* **Smart Filtering:** Isolate faulty stations by specific anomaly type or geographic region.
* **Hardware-Agnostic UI:** Decoupled frontend architecture designed for instant integration with backend ML data pipelines.

## Tech Stack
* **Frontend:** Streamlit, Pandas, Plotly
* **Backend:** Python, Scikit-learn (ML modeling for 1.8M row dataset)

## How to Run Locally
1. Clone the repository: `git clone https://github.com/yourusername/sih-skyguard.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the dashboard: `streamlit run app.py`

## Live Demo
vaayudrishti.streamlit.app
