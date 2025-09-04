# spotify-recommender
Python + Streamlit app recommending similar songs using cosine similarity and genre weighting
## Features
- Input a song → get 5 similar songs recommended  
- Data cleaning with pandas  
- Cosine similarity on tempo, valence, danceability, energy  
- Weighted genre matching (60% genre / 40% features)  
- First deployed project using Streamlit  

## Run locally or in a python termial
pip install -r requirements.txt
streamlit run app.py

Inspiration

Inspired by Spotify’s Smart Shuffle — I wanted to combine my love of music with Python.
