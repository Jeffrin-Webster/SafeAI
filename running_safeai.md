# 🚀 Running SafeAI: Monitor AI in Real-Time!

Get **SafeAI** up and running in minutes to detect harmful AI behaviors like blackmail or hate speech! Follow these steps to launch the app and explore its real-time monitoring, interactive charts, and ethical AI features.

```bash
# 1. Clone or Download the SafeAI Project
# 📂 Grab the project folder from GitHub or unzip the shared archive
git clone https://github.com/Jeffrin-Webster/SafeAI.git
cd SafeAI

# 2. Set Up a Virtual Environment (Recommended)
# 🛠️ Isolate dependencies for a clean setup
python -m venv safeai_env
# Windows: 
source safeai_env\Scripts\activate
# macOS/Linux: 
source safeai_env/bin/activate
# ✅ You’ll see (safeai_env) in your terminal

# 3. Install Dependencies
# 📦 Install required packages (streamlit, transformers, pandas, etc.)
pip install -r requirements.txt
# ⏳ This may take a few minutes (~1-2 GB download for models)

# 4. Run the SafeAI App
# 🌟 Launch the Streamlit app to start monitoring AI outputs
streamlit run app.py
# 🎉 Open http://localhost:8501 in your browser

