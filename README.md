
# 🧺 Laundry Business Intelligence Dashboard

## 📌 Project Overview

This project analyzes customer reviews of laundry businesses to extract actionable business insights. Using Natural Language Processing (NLP) and topic modeling, the system identifies key service themes and evaluates business performance based on review metrics.

The goal is to help businesses understand customer sentiment and identify areas for improvement.

---

## 🎯 Objectives

* Analyze laundry business performance using ratings and reviews
* Identify high-performing geographic locations
* Extract customer sentiment and common feedback themes
* Rank businesses based on real-world performance indicators
* Present insights in a simple, visual dashboard

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** (for dashboard UI)
* **pandas** (data processing)
* **Plotly** (interactive maps & charts)
* **scikit-learn** (LDA topic modeling)
* **NLP preprocessing (regex, text normalization)**

---

## 📂 Project Structure

```
├── app.py                # Streamlit dashboard (main UI)
├── analysis_engine.py   # Data processing + NLP + scoring logic
├── requirements.txt
├── dataset_raw_from_webscraping.csv
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The dataset used in this project was obtained from Kaggle and is distributed under the MIT License.

**Source:** https://www.kaggle.com/datasets/andri04/laundry-places-from-5-biggest-citiesindonesia

The dataset contains publicly available customer review information for laundry businesses.



## ⚙️ Methodology

### 1. Data Cleaning

* Removed null and empty reviews
* Normalized multilingual text
* Applied Indonesian + English stopword filtering

### 2. Feature Extraction

* Built Document-Term Matrix using CountVectorizer
* Filtered low-frequency noise words

### 3. Topic Modeling

* Applied Latent Dirichlet Allocation (LDA)
* Extracted dominant customer themes

### 4. Business Performance Scoring

Businesses were ranked using a composite score based on:

* Review rating
* Review count

---

## 🔍 Key Customer Insights

### Theme 1 — Service Quality & Affordability

* Focus on: friendly service, speed, pricing, neatness, fragrance
* Indicates strong all-around performance

---

### Theme 2 — Mixed / Neutral Feedback

* Includes contrast words like “but”
* Suggests balanced or slightly critical experiences

---

### Theme 3 — General Positive Reviews

* Simple, consistent praise
* Focus on: good service, fast, neat, pleasant smell

---

### Theme 4 — Convenience & Features

* Highlights delivery, timing, usability
* Indicates importance of service accessibility

---

## ⚠️ Limitations

* Relies mainly on review data (no pricing or operational costs)
* Multilingual text may introduce noise in topic modeling
* Some businesses may have incomplete or inconsistent data

---

## 🚀 How to Run

### 1. Clone the repository

```
git clone <your-repo-link>
cd <repo-name>
```

2. Create virtual environment (recommended):

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the dashboard

```
streamlit run app.py
```

---

## 📌 Future Improvements

* Add sentiment scoring (positive/negative classification)
* Improve Indonesian NLP preprocessing
* Add pricing-based analysis
* Real-time data scraping integration
* Deploy as a hosted web application

---

## 💡 Final Note

This project demonstrates how **data science + simple visualization** can directly help small businesses:

* Choose better locations
* Understand customer expectations
* Stay ahead of competitors

