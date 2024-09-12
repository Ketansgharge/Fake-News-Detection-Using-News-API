import streamlit as st
import requests
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
# Load the pre-trained TF-IDF vectorizer and model
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# Function to fetch news data from India News API
def fetch_news(api_key, country='in', category='general', num_articles=1):
    url = f'https://newsapi.org/v2/top-headlines'
    params = {'country': country, 'category': category, 'apiKey': api_key, 'pageSize': num_articles}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        if articles:
            return articles[0]['title']
        else:
            
            return None
    else:
        st.error(f"Error: {response.status_code}")
        return None

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

if __name__ == '__main__':
    st.title('Fake News Classification App')
    st.subheader("Input the News content below")
    
    # Fetch news from India News API
    api_key = 'b647494796a7445082b34c0cdfd7deef'  # Replace with your News API key
    india_news = fetch_news(api_key, country='in', category='general')
    
    # Display the fetched news in the text area
    sentence = st.text_area("Enter your news content here", "", height=200)
    
    predict_btt = st.button("Predict")
    
    if predict_btt:

        prediction_class = fake_news(sentence)
        # print(prediction_class)

        if len(sentence.split()) <= 20:
            st.warning('Unreliable')
        
        else:
            
            if prediction_class ==[1]:
                st.success('Reliable')
            # elif prediction_class == [1]:
            else:
                st.warning('Unreliable')
        # else:
        #     st.error('Prediction failed')
