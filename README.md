# MovieRatingPredictor

A hybrid content-based movie recommender system using TF-IDF features, user profile vectors and Ridge Regression.

## 🔧 Features
- TF-IDF features from genres & tags  
- Numeric movie metadata (popularity, mean rating, year, etc.)  
- User profile vectors (liked-movie centroid)  
- Ridge Regression rating predictor  
- Jupyter Notebook UI with TMDB posters & overviews  
- Cold-start handling and interactive refinement  

## 📦 Pipeline
1. **build_movie_features.py** - Create TF-IDF + numeric movie feature store  
2. **movielens_to_interactions.py** - Build interactions dataset  
3. **train_model.py** - Train Ridge Regression model  
4. **explain_model.py** - Show feature importances  
5. **recommend_for_new_user.py** - Generate personalized recommendations  
6. **movie-rating-prediction_pipeline.ipynb** - Full UI + workflow

## ▶️ Basic Usage
Follow the steps in the Jupyter Notebook file.

## 🧰 Requirements
- PIP Packages to be installed from *requirements.txt*
- TMDB API key in *.env* file