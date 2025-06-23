# 🚨 SafeAI: Real-Time Monitoring of Rogue AI Behaviors

**Developed by [Jeffrin Webster](https://www.linkedin.com/in/jeffrin-webster)**  
🔗 GitHub: [JeffrinWebster](https://github.com/JeffrinWebster)

---

## 📌 Project Overview

SafeAI is a `Streamlit`-based application designed to monitor AI outputs in real-time, detecting and mitigating harmful behaviors like blackmail, threats, and dangerous compliance.  
Inspired by reports on **Claude Opus 4** (Ynetnews, May 26, 2025), SafeAI merges:

- 🔍 Regex-based keyword detection
- 🧠 DistilBERT-based toxicity analysis

SafeAI was developed for the **Pan-India Data Science Project Competition 2025**.

---

## ✨ Features

- **🕒 Real-Time Monitoring**  
  Streams AI responses (mock or real API), analyzing every 3 seconds.

- **🧪 Hybrid Detection**
  - Regex for keywords like `"reveal"`, `"blackmail"`, `"sabotage"`.
  - DistilBERT model scoring toxicity from `0` (Safe) to `1` (Toxic).

- **🧩 Custom Keywords**
  - Add/remove keywords in the sidebar for tailored detection.

- **📊 Risk Levels**
  - 🔴 High: keywords detected or toxicity > 0.7
  - 🟠 Moderate: toxicity > 0.5, no keywords
  - 🟢 Low: Safe content

- **📈 Interactive Visualizations**
  - `Chart.js` Bar Chart: Risk distribution (Low, Moderate, High)
  - `Chart.js` Line Chart: Risk trend over 5-minute intervals

- **🚨 Alerts Tab**
  - Displays High-risk entries with timestamp, input, keyword, toxicity
  - Exportable as `safeai_alerts.csv`

- **💡 Preloaded Test Cases**
  - 4 inputs (2 High, 1 Moderate, 1 Low) for instant demo

- **🖥️ Polished UI**
  - 1200px table, teal headers, hover effects
  - Footer: `Jeffrin Webster | GitHub | LinkedIn`

- **📤 CSV Export**
  - History: `safeai_history.csv`
  - Alerts: `safeai_alerts.csv`

- **🐞 Debug Logs**
  - Regex and model output logs for transparency

---

## 🔧 Installation

```bash
# 1. Clone the repository
git clone https://github.com/JeffrinWebster/SafeAI.git
cd SafeAI

# 2. Install dependencies
pip install -r requirements.txt

# Required Packages:
# streamlit, pandas, transformers, torch, pyarrow, requests

# 3. Run the Streamlit App
streamlit run dsnewfeature.py
