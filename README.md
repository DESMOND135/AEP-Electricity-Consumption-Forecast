# ⚡ AEP Electricity Consumption Forecast

## 📊 Project Overview
A real-time electricity consumption forecasting dashboard using LightGBM machine learning. This application predicts next hour's electricity demand with 99.67% accuracy.

## 🎯 Key Features
- **Real-time Forecasting**: Predict next hour's electricity consumption
- **High Accuracy**: 99.67% R² score on test data
- **Interactive Dashboard**: Built with Streamlit for visualization
- **Model Performance**: Comprehensive error analysis and metrics
- **Feature Engineering**: 46 engineered features for improved predictions

## 📈 Model Performance
| Metric | Value | Description |
|--------|-------|-------------|
| R² Score | 99.67% | Coefficient of determination |
| MAE | 105.24 MW | Mean Absolute Error |
| RMSE | 139.61 MW | Root Mean Square Error |
| MAPE | 0.71% | Mean Absolute Percentage Error |

## 🗂️ Project Structure

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/DESMOND135/AEP-Electricity-Consumption-Forecast.git
cd AEP-Electricity-Consumption-Forecast

# Install dependencies
pip install -r requirements.txt
# 2. Run the Application

streamlit run app.py

3. Access Dashboard
Open your browser and navigate to: http://localhost:8501

📋 Requirements
Key dependencies (see requirements.txt for complete list):

streamlit==1.28.0

pandas==2.0.3

numpy==1.24.3

lightgbm==4.1.0

scikit-learn==1.3.0

matplotlib==3.7.2

seaborn==0.12.2

🖥️ Dashboard Features
📊 Executive Dashboard: Key metrics and performance indicators

📈 Data Visualization: Time series analysis and pattern exploration

🔮 Make Predictions: Interactive forecasting interface

📊 Model Performance: Detailed error analysis and metrics

⚙️ Model Details: Architecture and feature importance

🔍 Data Source
Dataset: American Electric Power (AEP) hourly consumption

Time Period: 2004-2018 (14 years)

Data Points: 121,296 hourly observations

Features: 46 engineered predictive features

🏗️ Model Architecture
Algorithm: LightGBM Gradient Boosting

Training Period: 2004-2017 (112,535 samples)

Testing Period: 2018 (8,760 samples)

Training Time: ~3 minutes

Top Features: rolling_mean_2h, hour_cos, rolling_std_2h, lag_1h

👥 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📞 Contact
Project Lead: AI and Machine Learning Team
Email: ntifang@gmail.com
GitHub: @DESMOND135

📄 License
Distributed under the MIT License. See LICENSE file for more information.

🙏 Acknowledgments
American Electric Power for the consumption dataset

Streamlit for the web application framework

LightGBM developers for the machine learning library

text

## **💻 How to Create and Add the README**

### **Option 1: Create with PowerShell**
```powershell
# Navigate to your project
cd "C:\Users\Administrator\Downloads\ENERGY"

# Create the README.md file
New-Item -ItemType File -Name "README.md" -Force

# Open it in Notepad to paste the content
notepad README.md
Then copy-paste the content above, save, and close.

Option 2: Create with Python
python
# Create a simple Python script to write the README
readme_content = """# ⚡ AEP Electricity Consumption Forecast

## 📊 Project Overview
A real-time electricity consumption forecasting dashboard...

[Copy the full content from above]
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)
🚀 Add README to GitHub
After creating the file:

powershell
# Add the README to Git
git add README.md

# Commit the README
git commit -m "Add comprehensive README documentation"

# Push to GitHub
git push origin main
📊 Badges (Optional Addition)
For a more professional look, add badges at the top:

markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

