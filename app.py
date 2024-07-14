import streamlit as st
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import gdown
import os

# Function to download model if not already downloaded
def download_model():
    model_file_path = 'twitter_sentiment.pkl'
    if not os.path.exists(model_file_path):
        file_id = '11pmJbbnbC-hL3qVVH3xJkQGHfkdnZS-N'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_file_path, quiet=False)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
    }
    .sidebar .sidebar-content {
        background-color: #E0E0E0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar content
st.sidebar.title("About the App")
st.sidebar.write("""
    This app performs Sentiment Analysis on tweets using a pre-trained model.
    You can visualize data distributions and word clouds based on the sentiment of tweets.
""")

st.sidebar.title("Navigation")

# Navigation buttons
if st.sidebar.button("Home"):
    st.session_state.page = "home"
if st.sidebar.button("Data Visualization"):
    st.session_state.page = "visualization"
if st.sidebar.button("Prediction"):
    st.session_state.page = "prediction"
if st.sidebar.button("About"):
    st.session_state.page = "about"

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Main content
if st.session_state.page == "home":
    st.title('Twitter Sentiment Analysis')
    
    # Add sentiment.png to the header
    st.image("sentiment.png", use_column_width=True)

    st.write("""
    Welcome to the Twitter Sentiment Analysis app. Use the sidebar to navigate through different sections.
    """)

elif st.session_state.page == "visualization":
    st.header("Data Visualization")
    df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_sentiment.csv', header=None, index_col=[0])
    df = df[[2,3]].reset_index(drop=True)
    df.columns = ['sentiment', 'text']
    df.dropna(inplace=True)
    df = df[df['text'].apply(len)>1]

    # 3D scatter plot for sentiment analysis
    st.subheader("3D Scatter Plot for Sentiment Analysis")
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))

    fig = px.scatter_3d(df, x='text_length', y='word_count', z='sentiment',
                        color='sentiment', symbol='sentiment',
                        title="3D Scatter Plot of Tweet Length, Word Count, and Sentiment")
    st.plotly_chart(fig)

    # Pie chart of sentiment distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    df['sentiment'].value_counts().plot(kind='pie', autopct='%1.0f%%', ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

    # Word cloud for each sentiment
    st.subheader("Word Clouds by Sentiment")
    stopwords = set(STOPWORDS)
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    for index, sentiment in enumerate(df['sentiment'].unique()):
        ax = axes[index//2, index%2]
        sentiment_data = df[df['sentiment']==sentiment]['text']
        wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=500, max_font_size=40, scale=5).generate(str(sentiment_data))
        ax.imshow(wordcloud)
        ax.set_title(sentiment, fontsize=20)
        ax.axis('off')

    st.pyplot(fig)

elif st.session_state.page == "prediction":
    st.header("Prediction")

    tweet = st.text_input('Enter your tweet')
    submit = st.button('Predict')

    if submit:
        # Download the model if not already downloaded
        download_model()

        # Load the model
        model_file_path = 'twitter_sentiment.pkl'
        if os.path.exists(model_file_path):
            model = pickle.load(open(model_file_path, 'rb'))

            # Perform prediction
            start = time.time()
            prediction = model.predict([tweet])
            end = time.time()

            # Display prediction results
            st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
            st.write('Sentiment:', prediction[0])
        else:
            st.write("Error: Model file not found. Please check your connection or try again later.")

elif st.session_state.page == "about":
    st.header("About Twitter Sentiment Analysis")
    st.write("""
        Twitter Sentiment Analysis is a technique used to determine the sentiment of tweets. 
        Sentiment analysis involves the use of natural language processing (NLP) and machine learning to classify text into 
        different sentiments such as positive, negative, or neutral.

        **Methods used in this application:**
        - **Data Cleaning**: Removing noise from the data such as URLs, HTML tags, and special characters.
        - **Feature Extraction**: Converting text data into numerical features using techniques like TF-IDF.
        - **Model Building**: Using machine learning models like Random Forest Classifier to train on the extracted features.
        - **Model Evaluation**: Evaluating the performance of the model using metrics like classification report.
        - **Data Visualization**: Visualizing the data distribution and important features using histograms and word clouds.

        **Benefits and Advantages of Sentiment Analysis:**
        - **Business Insights**: Helps businesses understand customer opinions and feedback.
        - **Market Research**: Analyzes market trends and public sentiment towards products or services.
        - **Customer Service**: Identifies and addresses customer complaints and issues promptly.
        - **Decision Making**: Provides valuable insights for data-driven decision making.
    """)
