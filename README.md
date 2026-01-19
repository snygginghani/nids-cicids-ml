# 🛡️ Network Intrusion Detection System (NIDS) – CIC-IDS2017

An **end-to-end Machine Learning + Cybersecurity project** that detects malicious
network traffic using supervised learning and anomaly detection, with an
interactive **web dashboard**.

---

## 🚀 Features

- Flow-based network traffic analysis using **CIC-IDS2017**
- Binary classification: **Benign vs Attack**
- Multiple attack categories:
  - DDoS
  - DoS
  - PortScan
  - Botnet
  - Brute Force
  - Web Attacks
- Machine Learning models:
  - Random Forest
  - XGBoost
  - Isolation Forest (anomaly detection)
- Interactive **Streamlit web application**
- Realistic **end-to-end ML pipeline**

---

## 🧠 Architecture

Parquet Data → Preprocessing → ML Models → Saved Models → Web Dashboard


---

## 📂 Project Structure

nids-cicids-ml/
├─ data/ # CIC-IDS2017 parquet files (not uploaded)
├─ models/ # Trained ML models
├─ config.py
├─ data_utils.py
├─ preprocess.py
├─ main_supervised.py # Train RandomForest & XGBoost
├─ main_anomaly.py # Train IsolationForest
├─ stream_simulator.py # Simulated live traffic
├─ web_app.py # Streamlit web dashboard
├─ requirements.txt
├─ RF_Binary_cm.png
├─ XGB_Binary_cm.png
└─ README.md


---

## 📊 Dataset

**CIC-IDS2017 – Canadian Institute for Cybersecurity**

- Official site:  
  https://www.unb.ca/cic/datasets/ids-2017.html
- Kaggle mirror (parquet format available)

Place the downloaded `.parquet` files inside:

data/


---

## ⚙️ Installation

```bash
pip install -r requirements.txt
🏋️ Train Models
python main_supervised.py
python main_anomaly.py
🌐 Run Web Application
streamlit run web_app.py
or (normal Python execution):

python run_web_app.py
📈 Output
Classification reports

Confusion matrices

Attack vs benign distribution charts

Downloadable CSV predictions

Interactive dashboard for traffic analysis

🎯 Use Cases
SOC monitoring simulation

Intrusion detection research

Cybersecurity & ML portfolio project

Graduation / academic project

👤 Author
Hani Muhannad
Machine Learning & Data Analytics
University of Jordan


---

### ✅ After you paste it:
1. Click **Commit changes**
2. Refresh the repo page
3. Your project now looks **100% professional**

If you want, next I can:
- Write a **short project explanation** for your CV
- Or a **LinkedIn post** announcing the project
- Or help you **dockerize** it for extra points 🚀
