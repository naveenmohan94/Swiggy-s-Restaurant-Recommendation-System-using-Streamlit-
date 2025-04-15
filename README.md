# 🧠 Mini Project 4: Swiggy’s Restaurant Recommendation System using Streamlit

## 🚀 Project Overview
This project is a personalized restaurant recommendation system built using Python and Streamlit. It processes a large Swiggy dataset, performs data cleaning, encoding, dimensionality reduction (PCA), clustering (KMeans), and enables intelligent, real-time recommendations through a sleek web UI. Users can filter restaurants based on **city**, **cuisine**, **rating**, **reviews**, **cost**, and **distance metric (Euclidean/Cosine)**.

---

## 🧠 Skills & Concepts Demonstrated
- ✅ Data Cleaning & Preprocessing  
- ✅ One-Hot Encoding of Categorical Variables  
- ✅ PCA Dimensionality Reduction  
- ✅ K-Means Clustering  
- ✅ Cosine Similarity & Euclidean Distance  
- ✅ Streamlit App Development  
- ✅ Model Serialization with Pickle  
- ✅ Git & Version Control  

---

## 📊 Tech Stack
- **Language**: Python  
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `pickle`, `streamlit`  
- **Interface**: Streamlit  
- **Domain**: Recommendation Systems, Data Analytics  

---

## 🧠 Problem Statement
Build an intelligent recommendation system that suggests restaurants based on user preferences like:
- City
- Cuisine
- Rating
- Rating Count
- Cost  
The system must:
- Be fast and scalable using clustering
- Have an interactive Streamlit front-end
- Allow distance-based filtering (Euclidean or Cosine)

---

## 📂 Dataset Overview

| File | Description |
|------|-------------|
| `swiggy.csv` | Raw dataset |
| `cleaned_data.csv` | Cleaned dataset with necessary columns |
| `encoded_data.csv` | One-hot encoded & scaled version |
| `pca_encoded_data.csv` | PCA-reduced features |
| `city_columns.csv`, `cuisine_columns.csv` | Encoded column references |
| `.pkl files` | Encoders, PCA, Scaler, and KMeans models |

---

## 🔍 Columns Explained
- `id`: Unique restaurant ID  
- `name`: Restaurant name  
- `city`: Restaurant location  
- `rating`: Average user rating (cleaned from "--")  
- `rating_count`: Number of user ratings (e.g. "1.2K" → 1200)  
- `cost`: Average cost for two (₹ removed)  
- `cuisine`: Multi-cuisine information  
- Dropped: `lic_no`, `link`, `address`, `menu`  

---

## 𞶙 Data Preprocessing
- Replaced missing/invalid ratings (`--`) with 4.0  
- Converted rating counts like "50+" or "1.2K" into integers  
- Cleaned cost values by removing currency symbols  
- Handled missing values in `name`, `rating`, `rating_count`, `cost` with `groupby('city')` + `mode/median`  
- Dropped records missing crucial data like `cuisine`  
- Shape reduced from **(148541, 11)** to **(148299, 11)**  

---

## 📊 Exploratory Data Analysis
- **Top 20 Restaurant Chains**: Domino's, Pizza Hut, KFC, etc.  
- **Cuisine Frequency**: Most popular - Indian, Chinese; Rare - Khasi, Gujarati-Italian  
- **Most Expensive Cuisines**: Sri Lankan, Japanese-Mughlai, European  

---

## 🧬 Encoding & Feature Transformation
- `cuisine`: Encoded using `MultiLabelBinarizer`  
- `city`: Encoded using `OneHotEncoder`  
- Numeric columns (`rating`, `rating_count`, `cost`) scaled using `StandardScaler`  
- Dimensionality reduced to **30 components** using `PCA`  

---

## ⚙️ Modeling
- Used **KMeans** clustering with `10 clusters`  
- Applied **Cosine Similarity** & **Euclidean Distance** for matching  
- Saved all essential models using `pickle`:
  - `cuisine_encoder.pkl`
  - `city_encoder.pkl`
  - `scaler.pkl`
  - `pca_model.pkl`
  - `kmeans_model.pkl`
  - `pca_input_columns.pkl`

---

## 🤖 Recommendation Logic
1. User selects:
   - City
   - Cuisine
   - Rating, Rating Count, Cost
   - Distance Metric (Euclidean / Cosine)
2. Input encoded & scaled → PCA transformed
3. Predicted KMeans cluster
4. Restaurants filtered by same cluster + city + cuisine
5. Ranked using distance metric
6. **Top 10** recommendations displayed with:
   - Restaurant name
   - Rating, cost, cuisine
   - Google Maps link & order page

---

## 🖥️ Streamlit UI Features
- Sidebar filters with dynamic dropdowns  
- Stylish colored headers & user-friendly layout  
- Integrated external links for Google Maps & ordering  
- Uses minimal memory by avoiding reloading huge datasets  

---

## 📌 Business Use Cases
- Personalized UX for customers  
- Enhanced filtering for food delivery platforms  
- Market trend analysis (popular cuisines, cities, etc.)  
- Scalable for real-world applications  

---

## 📈 Results
- Real-time, accurate recommendations  
- Handles large datasets efficiently  
- Combines clustering with similarity metrics for precision  
- Intuitive Streamlit interface with visual appeal

---

## 📂 Directory Structure
```bash
📆swiggy-recommendation
 ├️ 📆models
 │  ├️ city_encoder.pkl
 │  ├️ cuisine_encoder.pkl
 │  ├️ scaler.pkl
 │  ├️ pca_model.pkl
 │  └️ kmeans_model.pkl
 ├️ 📆data
 │  ├️ cleaned_data.csv
 │  ├️ pca_encoded_data.csv
 │  └️ pca_input_columns.pkl
 ├️ app.py
 ├️ README.md
 └️ requirements.txt
```

