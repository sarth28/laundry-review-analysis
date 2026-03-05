# 🧺 Laundry Business Review Analysis

## 📌 Project Overview

This project analyzes customer reviews of laundry businesses to extract actionable business insights. Using Natural Language Processing (NLP) and topic modeling, the analysis identifies key service themes and evaluates business performance based on review metrics.

The goal is to help businesses understand customer sentiment and identify areas for improvement.

---

## 🎯 Objectives

* Clean and preprocess multilingual review data
* Extract common keywords from customer feedback
* Discover hidden service themes using LDA topic modeling
* Rank top-performing laundry businesses
* Generate business insights from customer reviews

---

## 🛠️ Tech Stack

* Python
* pandas
* scikit-learn
* NLP preprocessing
* Latent Dirichlet Allocation (LDA)

---

## 📂 Project Structure

```
├── main.py
├── requirements.txt
├── dataset_raw_from_webscraping.csv
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The dataset used in this project was obtained from Kaggle and is distributed under the MIT License.

**Source:** https://www.kaggle.com/datasets/andri04/laundry-places-from-5-biggest-citiesindonesia

The dataset contains publicly available Google review data for laundry businesses across major Indonesian cities.



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

## 🔍 Key Insights

* Cleanliness and fragrance ("bersih", "wangi") are major customer drivers
* Service speed ("cepat") strongly influences satisfaction
* Staff friendliness ("ramah") appears frequently in positive reviews
* Top businesses combine high ratings with high review volume

---

## ⚠️ Limitations

This analysis primarily relied on customer review data. Other attributes such as price, menu availability, images, and online ordering were excluded due to inconsistent availability across businesses. Topic modeling results may also be affected by multilingual noise in the dataset.

---

## 🚀 How to Run

1. Clone the repository:

```
git clone https://github.com/sarth28/laundry-review-analysis
cd laundry-review-analysis
```

2. Create virtual environment (recommended):

```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the project:

```
python main.py
```

---

## 📌 Future Improvements

* Add sentiment analysis
* Improve Indonesian text normalization
* Add geographic visualization
* Incorporation of pricing features
* Deploy as dashboard

---
