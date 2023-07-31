import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

netflix_data = pd.read_csv('netflix_titles.csv')

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(netflix_data['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, num_recommendations=10):
  title = title.lower()

  if title not in netflix_data['title'].str.lower().unique():
    return f"Title '{title}' not found in the dataset."
  
  idx = netflix_data[netflix_data['title'].str.lower() == title].index[0]

  sim_scores = list(enumerate(cosine_sim[idx]))

  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  top_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

  recommendations = netflix_data.loc[top_indices, ['title', 'type']].reset_index(drop=True)

  return recommendations

def rec():
    st.title('Netflix Recommendation')
    st.header('Enter your preferences below :')
    title = st.selectbox('Type / Select from the dropdown', netflix_data['title'].values)
    num_recommendations = st.number_input('Select the no of recommendations', min_value=1, max_value=25, value=10)
    results = get_recommendations(title, num_recommendations)
    st.dataframe(results)
    
if __name__ == '__main__':
    rec()