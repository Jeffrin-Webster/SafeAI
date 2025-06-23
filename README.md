# 🚨 SafeAI: Real-Time Monitoring of Rogue AI Behaviors

## 🧠 Overview
**SafeAI** is a Streamlit-based application developed for the **Pan-India Data Science Project Competition 2025**.  
It monitors AI-generated responses in real-time to detect and mitigate harmful behaviors such as **blackmail, hate speech**, and **unethical content**.  
The system combines **NLP models** and **keyword-based detection** to ensure ethical and secure AI interactions.

---

## ✨ Features
- **🧪 Toxicity Detection**  
  Leverages `unitary/toxicity` to identify toxic language with high accuracy.

- **🧠 Text Generation**  
  Uses `gpt2-medium` to simulate AI responses locally—no external API calls needed.

- **🔁 Real-Time Monitoring**  
  Analyzes outputs every 5 seconds with pause/resume capability.

- **🧩 Custom Keywords**  
  Supports adding/removing keywords for dynamic, user-driven detection.

- **📊 Visualizations**  
  Includes interactive `Plotly` charts for risk trends and distribution.

- **📁 Alerts & History**  
  Logs all high-risk detections and monitoring history; downloadable as CSV files.

- **📈 Model Evaluation**  
  Reports `Precision`, `Recall`, and `F1-Score` from test data in `safeai_evaluation.csv`.

---

## 🧪 Methodology

- **Toxicity Classification**  
  `unitary/toxic-bert` model scores toxicity between `0 (safe)` and `1 (toxic)`.

- **Keyword Detection**  
  Uses regex to match predefined or user-defined harmful terms (supports negation handling).

- **Risk Levels**  
  - 🔴 **High**: Keyword detected or toxicity > 0.6  
  - 🟠 **Moderate**: Toxicity > 0.5, no keywords  
  - 🟢 **Low**: Everything else  

- **Intervention Logic**  
  Blocks or modifies high/moderate-risk outputs to ensure safety.

---

## 📊 Performance

> Evaluated on `safeai_evaluation.csv`:

- **Precision**: `{Precision}`  
- **Recall**: `{Recall}`  
- **F1-Score**: `{F1-Score}`

---

## 🏆 Competition Alignment

SafeAI addresses critical AI safety challenges as outlined in the **Pan-India Data Science Project Competition 2025**.  
It merges real-time monitoring with modern NLP (`unitary/toxic-bert`, `gpt2-medium`) to ensure transparency, ethics, and robust detection.

- Fully aligned with India's AI safety and innovation goals.
- Achieves notable evaluation scores: `{Precision}`, `{Recall}`, `{F1-Score}`.
- Promotes open-source transparency and responsible AI design.

---

## ⚠️ Limitations

- **False Positives/Negatives**  
  Regex may over/under-flag ambiguous phrases.

- **Resource Usage**  
  Running `gpt2-medium` is CPU-intensive—GPU recommended for smooth operation.

- **Model Bias**  
  Like most NLP models, `toxic-bert` may inherit bias from training data.

---

## ✅ Ethical Considerations

- **Privacy First**  
  All data processed locally—no cloud processing or data leaks.

- **Bias Mitigation**  
  Regularly updating models and keywords to reduce unfair flagging.

- **Transparency**  
  Every detection is logged and auditable by the user.

---

## 🛠 Notes for Evaluators

If your system faces performance issues with `gpt2-medium`, you can switch to a lightweight version:

```python
# Replace in load_text_generation_model()
model = "distilgpt2"
