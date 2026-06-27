# 🇳🇬 Nigeria Exchange Rate Anomaly Detector

A data engineering and machine learning dashboard that detects anomalies in the USD/NGN exchange rate using three detection methods: **Isolation Forest**, **Z-Score**, and **Rolling Deviation**.

## Live Demo

[View on Streamlit Cloud](https://nigeria-anomaly-detector.streamlit.app/)


## What it does
- Pulls historical CBN USD/NGN exchange rate data
- Runs three anomaly detection algorithms in parallel
- Flags and labels major economic shock events (2016 devaluation, 2020 COVID crash, 2023 naira float)
- Visualizes anomalies interactively with hover context and % change
- Allows date filtering and CSV export of flagged anomalies

## Key Events Detected
| Date | Event | Rate Change |
|------|-------|-------------|
| Jul 2016 | CBN devaluation | ₦197 → ₦305 (+55%) |
| Jul 2020 | COVID-19 crash | ₦360 → ₦461 (+28%) |
| Jun 2023 | Tinubu naira float | ₦463 → ₦770 (+66%) |
| Dec 2023 | Further depreciation | ₦1,490 |

## Tech Stack
- **Python** — data pipeline and ML
- **Streamlit** — dashboard and deployment
- **Scikit-learn** — Isolation Forest
- **Plotly** — interactive charts
- **Pandas/NumPy** — data wrangling

## Run locally
```bash
git clone https://github.com/The-Fugazi/nigeria-anomaly-detector
cd nigeria-anomaly-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Author
**Allison Oluwatobi Samuel** — Data Analyst & ML Engineer
[GitHub](https://github.com/The-Fugazi) · [LinkedIn](https://www.linkedin.com/in/allison-oluwatobi/)
