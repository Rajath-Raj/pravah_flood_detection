# 🌊 Pravah - Flood Detection Project

> **Pravah** (प्रवाह) - word meaning "flow" or "current"

## 📋 Project Overview

**Pravah** is a beginner-friendly machine learning project for learning data science fundamentals. Pravah is a beginner-friendly machine learning project for learning data science fundamentals. This project focuses on building a binary classification model to predict the severity of a flood event using a range of geographical and meteorological features.

🎯 Learning Goals

The primary goal is to build a robust data pipeline that prepares the dataset for a binary classification task. The key learning objectives achieved in this project include:

Exploratory Data Analysis (EDA): Performing a thorough analysis of the dataset to understand feature distributions, identify anomalies (such as the high number of zero values in the distance column), and examine correlations between variables.

Data Cleaning & Preprocessing: Identifying and handling columns that could lead to data leakage (Total Deaths, Total Affected) and preparing the data for modeling.

Feature Engineering: Creating a new, meaningful target variable (Is_Severe_Flood) from existing features. This was the critical step that enabled the project to move forward.

Model Preparation: Implementing a standard machine learning workflow, including splitting the data into training and testing sets and applying feature scaling with StandardScaler.

Library Proficiency: Gaining hands-on experience with core Python libraries: Pandas for data manipulation, NumPy for numerical operations, Matplotlib/Seaborn for visualization, and Scikit-learn for preprocessing and modeling.


## 🏗️ Current Project Structure

```
pravah_flood_detection/
├── 📊 flood_data_exploration.ipynb    # Data exploration notebook (IN PROGRESS)
├── 📦 requirements.txt                 # Python dependencies  
├── 📚 README.md                        # Project documentation
└── ⚙️ venv/                           # Virtual environment
```

## � What I've Done So Far

### ✅ **Environment Setup** 
- Created virtual environment (`venv`)
- Installed required Python libraries
- Set up Jupyter notebook for data exploration

### ✅ **Data Exploration (Current Phase)**
- Basic data loading with pandas
- Dataset shape and structure analysis
- Statistical summaries using `describe()` and `info()`
- Missing value detection with `isna().sum()`
- Duplicate row checking

## 📊 Current Progress

**Phase 1: Data Exploration** 🔄 **(Currently Working On)**
- ✅ Load dataset with pandas
- ✅ Check dataset shape and basic info
- ✅ Identify missing values
- ✅ Check for duplicates
- 🔄 Explore feature distributions
- ⏳ Visualize data patterns

**Phase 2: Data Cleaning** ⏳ **(Next Steps)**
- Handle missing values
- Remove or fix duplicates
- Basic data preprocessing

**Phase 3: Model Building** ⏳ **(Future)**
- Split data into training/testing
- Train a simple classification model
- Evaluate model performance

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Basic knowledge of Python

### What I Did to Set Up

1. **Cloned the repository**
   ```bash
   git clone https://github.com/Rajath-Raj/pravah_flood_detection.git
   cd pravah_flood_detection
   ```

2. **Created virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

3. **Installed basic libraries**
   ```bash
   pip install -r requirements.txt
   ```

4. **Started Jupyter notebook**
   ```bash
   jupyter notebook flood_data_exploration.ipynb
   ```

## 📦 Libraries I'm Learning

| Library | What I'm Using It For |
|---------|----------------------|
| `pandas` | Loading and exploring CSV data |
| `numpy` | Basic numerical operations |
| `matplotlib` | Creating simple plots (coming soon) |
| `seaborn` | Making prettier visualizations (coming soon) |
| `jupyter` | Interactive data exploration |






### Next Steps
- [ ] Add some basic visualizations
- [ ] Understand the dataset better
- [ ] Learn about different data types
- [ ] Start basic data cleaning

## 🤝 Learning Resources

Since I'm just starting out, these resources are helpful:
- [Pandas Documentation](https://pandas.pydata.org/docs/user_guide/)
- [Jupyter Notebook Basics](https://jupyter.org/try)
- [Python for Data Science](https://www.python.org/about/gettingstarted/)

## 📞 Contact

**Rajath Raj**
- GitHub: [@Rajath-Raj](https://github.com/Rajath-Raj)
- Project Link: [https://github.com/Rajath-Raj/pravah_flood_detection](https://github.com/Rajath-Raj/pravah_flood_detection)

---

🚀 **This is a learning project - progress updates coming as I learn more!** 📚

*Learning data science one step at a time* ✨
