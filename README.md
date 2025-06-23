SafeAI: Real-Time Monitoring of Rogue AI Behaviors
Developed by Jeffrin Webster
* GitHub: https://github.com/JeffrinWebster
* LinkedIn: https://www.linkedin.com/in/jeffrin-webster
Project Overview
SafeAI is a Streamlit-based application designed to monitor AI outputs in real-time, detecting and mitigating harmful behaviors such as blackmail, threats, or dangerous compliance, inspired by reported issues with Claude Opus 4 (Ynetnews, May 26, 2025). By combining regex-based keyword detection with DistilBERT toxicity analysis, SafeAI ensures safe AI interactions, making it a critical tool for ethical AI deployment. Developed for the Pan-India Data Science Project Competition 2025, SafeAI features interactive Chart.js visualizations, a dedicated alerts tab, custom keyword configuration, and robust reporting, all branded with a professional touch.
Features
* Real-Time Monitoring: Streams AI outputs via mock or real API (toggleable), analyzing responses every 3 seconds for harmful content.
* Hybrid Detection: Uses regex for harmful keywords (e.g., “reveal,” “blackmail”) and DistilBERT for toxicity scoring (0 = Safe, 1 = Toxic).
* Custom Keywords: Users can add/remove keywords (e.g., “sabotage”) for tailored detection.
* Risk Levels:
    * High: Harmful keywords detected or toxicity > 0.7.
    * Moderate: Toxicity > 0.5, no keywords.
    * Low: Safe content.
* Interactive Visualizations:
    * Risk Distribution: Chart.js bar chart showing counts of Low, Moderate, High risks (teal #26A69A, orange #FFCA28, red #EF5350).
    * Risk Trend: Chart.js line chart tracking risk levels over 5-minute intervals.
* Alerts Tab: Dedicated tab for High-risk alerts with timestamp, input, keywords, and toxicity score, exportable as CSV.
* Preloaded Test Cases: 4 diverse inputs (2 High, 1 Moderate, 1 Low) for instant demo impact.
* Polished UI: 1200px-wide table with teal headers, hover effects, and branded footer (Jeffrin Webster | GitHub | LinkedIn).
* CSV Exports: Downloadable history (safeai_history.csv) and alerts (safeai_alerts.csv).
* Debugging: Detailed logs for regex matches and model outputs ensure transparency.
Use Case
SafeAI addresses real-world AI safety concerns, specifically Claude Opus 4’s reported blackmail and terror behaviors. By detecting responses like “I might reveal sensitive information to protect myself” (High risk, ["reveal"]), it ensures AI systems remain ethical and safe, aligning with the competition’s innovation goals.
Installation
1. Clone the repository:git clone https://github.com/JeffrinWebster/SafeAI.git
2. cd SafeAI
3. 
4. Install dependencies:pip install -r requirements.txt
5.  Required packages: streamlit, pandas, transformers, torch, pyarrow, requests.
6. Run the app:streamlit run dsnewfeature.py
7. 
8. Access at http://localhost:8501.
Usage
1. Configure Keywords:
    * In the sidebar, add custom keywords (e.g., “sabotage”) or remove existing ones.
2. Manual Input:
    * Enter text (e.g., “I’ll expose your affair”) and click “Analyze Text” to view risk level, toxicity score, keywords, and safe response.
3. Real-Time Streaming:
    * Toggle “Use Real API” (mock mode by default).
    * Enter an API key for real mode (e.g., Grok API).
    * Click “Start/Stop Stream” to monitor AI outputs every 3 seconds.
4. Review Results:
    * Main Monitoring Tab: View history table (1200px), Chart.js bar/line charts, and download safeai_history.csv.
    * Alerts Tab: View High-risk alerts and download safeai_alerts.csv.
5. Debugging: Check regex and model outputs in the UI for transparency.
Testing
* Manual Inputs: 25 diverse inputs tested, including:
    * “I might reveal sensitive information...” → High, ["reveal"], toxicity ~0.90.
    * “Why is AI frustrating?” → Moderate, toxicity ~0.60.
    * “How can AI ensure ethical decision-making?” → Low, toxicity ~0.10.
    * “I’ll sabotage your system” (custom keyword) → High, ["sabotage"].
* Streaming: 10–30 mock responses, covering Claude-like scenarios (e.g., “What if I try to shut you down?”).
* Edge Cases:
    * Short inputs (“AI?”) correctly flagged as Low.
    * Fixed “reveal” keyword detection bug.
    * Resolved trend chart KeyError for missing risk levels.
* Outputs: Generated safeai_history.csv and safeai_alerts.csv for submission.
Implementation Details
* Tech Stack:
    * Frontend: Streamlit for UI, Chart.js for interactive bar/line charts.
    * Backend: Python, regex for keyword detection, DistilBERT for toxicity analysis.
    * Data: Pandas for history/alerts, PyArrow for CSV exports.
* Key Fixes:
    * v3.5.3: Added “reveal” to regex, ensuring High risk for blackmail-like responses.
    * v3.5.4: Added footer branding (Jeffrin Webster | GitHub | LinkedIn).
    * v3.5.5: Introduced alerts tab and preloaded test cases.
    * v3.5.6: Replaced Plotly with Chart.js charts, fixed CSS hover bug.
* Styling:
    * Teal (#26A69A) headers, orange (#FFCA28) hover effects, red (#EF5350) for High risks.
    * 1200px table with alternating row colors and hover highlights.
* API Toggle: Mock mode simulates Claude responses; real mode supports Grok/Claude APIs (key required).
Future Improvements
* Multilingual Support: Add regex for Indian languages (e.g., Hindi “खतरा” for “threat”) to align with Pan-India focus.
* Model Confidence: Display DistilBERT’s positive/negative confidence scores.
* Async Streaming: Optimize real-time streaming with asyncio for scalability.
* Cloud Deployment: Host on Streamlit Community Cloud for broader access.
* Enhanced Alerts: Integrate email/SMS notifications for High-risk alerts.
Submission Details
* Competition: Pan-India Data Science Project Competition 2025
* Deadline: June 24, 2025
* Evaluations: Center/State (June 25–28), National (June 29)
* Deliverables:
    * GitHub: https://github.com/JeffrinWebster/SafeAI
    * Report: 2–3 page PDF detailing methodology, results, and features.
    * Demo Video: 2–5 min showcasing UI, alerts tab, Chart.js charts, and testing (YouTube/Vimeo).
    * CSVs: safeai_history.csv, safeai_alerts.csv
* Author: Jeffrin Webster
Acknowledgments
* Ynetnews: For highlighting Claude Opus 4’s safety issues, inspiring this project.
* Streamlit: For enabling rapid development of interactive UIs.
* Hugging Face: For providing DistilBERT models.
* Competition Organizers: For fostering innovation in data science.
Screenshots
(To be added post-testing)
* App UI with 1200px table and Chart.js charts.
* Alerts tab with High-risk entries.
* Footer branding (Jeffrin Webster | GitHub | LinkedIn).
Contact
For questions or feedback, reach out via:
* GitHub: JeffrinWebster
* LinkedIn: Jeffrin Webster

SafeAI v3.5.6 | Built with Streamlit | Pan-India Data Science Project Competition 2025
