import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Course Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (minimized)
st.markdown("""
    <style>
    .main-header {font-size:36px;font-weight:bold;color:#1E88E5;text-align:center;margin-bottom:20px;text-shadow:2px 2px 4px rgba(0,0,0,0.1);}
    .sub-header {font-size:24px;font-weight:bold;color:#004D40;margin-top:20px;margin-bottom:10px;}
    .card {border-radius:10px;padding:20px;background-color:rgba(255,255,255,0.9);box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-bottom:20px;}
    .recommendation-card {background-color:#f1f8ff;border-left:5px solid #1E88E5;padding:15px;margin-bottom:10px;border-radius:5px;transition:transform 0.3s ease;}
    .recommendation-card:hover {transform:translateX(5px);box-shadow:0 4px 8px rgba(0,0,0,0.1);}
    .course-title {font-weight:bold;color:#0D47A1;}
    .course-rating {color:#FF8F00;font-weight:bold;}
    .similarity-score {color:#00695C;font-size:0.9rem;}
    .stTextInput>label {font-size:18px;font-weight:bold;color:#004D40;}
    .stSlider>label {font-size:16px;font-weight:bold;color:#004D40;}
    .metrics-container {display:flex;justify-content:space-between;margin-top:20px;}
    .metric-card {background-color:#e3f2fd;border-radius:10px;padding:15px;flex:1;margin:0 10px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
    .metric-value {font-size:24px;font-weight:bold;color:#0D47A1;}
    .metric-label {font-size:14px;color:#455A64;}
    </style>
    """, unsafe_allow_html=True)

# Load NLTK stopwords only once at startup
@st.cache_data
def load_stopwords():
    try:
        # Import inside function to avoid global import
        import nltk
        from nltk.corpus import stopwords
        try:
            return set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            return set(stopwords.words('english'))
    except ImportError:
        # Fallback to a small set of common stopwords if NLTK is not available
        return set(['the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                   'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 
                   'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 
                   'to', 'from', 'in', 'on', 'by', 'with', 'at'])

# Load spaCy only if needed (lazy loading)
@st.cache_resource
def get_nlp():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except:
            import os
            os.system("python -m spacy download en_core_web_sm")
            return spacy.load("en_core_web_sm")
    except ImportError:
        st.warning("spaCy not installed or model not found. Using simpler text processing.")
        return None

@st.cache_data
def load_data():
    try:
        # Try loading the dataset
        df = pd.read_csv("Final_Coursera.csv")
        return df
    except FileNotFoundError:
        # Create a sample dataset for debugging if the file is not found
        st.error("Dataset file 'Final_Coursera.csv' not found. Using a sample dataset for debugging.")
        
        # Sample data
        data = {
            'Course Name': [
                'Introduction to Machine Learning',
                'Python Programming for Beginners',
                'Data Science Fundamentals',
                'Web Development with JavaScript',
                'Deep Learning Specialization',
                'SQL for Data Analysis',
                'Mobile App Development with Flutter',
                'Cloud Computing Basics',
                'Cybersecurity Fundamentals',
                'Business Analytics'
            ],
            'Course Description': [
                'Learn the basics of machine learning algorithms and applications',
                'Start your programming journey with Python fundamentals',
                'Master the core concepts of data science with practical examples',
                'Build modern websites with JavaScript, HTML, and CSS',
                'Understand neural networks and their applications',
                'Use SQL to query databases and analyze data effectively',
                'Create cross-platform mobile apps with Flutter',
                'Introduction to cloud services and deployment',
                'Learn about security principles and threat prevention',
                'Apply data analysis to business problems'
            ],
            'Skills': [
                'Machine Learning, Python, Statistics',
                'Python, Programming Basics, Problem Solving',
                'Data Analysis, Statistics, Visualization',
                'JavaScript, HTML, CSS, Web Development',
                'Neural Networks, TensorFlow, AI',
                'SQL, Database Management, Data Analysis',
                'Flutter, Dart, Mobile Development',
                'AWS, Azure, Cloud Infrastructure',
                'Network Security, Encryption, Risk Management',
                'Excel, Data Visualization, Business Intelligence'
            ],
            'Course Rating': [
                4.7, 4.5, 4.8, 4.2, 4.9, 4.6, 4.3, 4.4, 4.5, 4.7
            ],
            'Course URL': [
                'https://example.com/ml',
                'https://example.com/python',
                'https://example.com/data-science',
                'https://example.com/web-dev',
                'https://example.com/deep-learning',
                'https://example.com/sql',
                'https://example.com/flutter',
                'https://example.com/cloud',
                'https://example.com/security',
                'https://example.com/analytics'
            ]
        }
        
        return pd.DataFrame(data)

def clean_text(text):
    """Optimized text cleaning function"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[&:_\-\(\)\+\'\.,]', ' ', text)
    
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@st.cache_data
def preprocess_data(df):
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Create tags column more efficiently
    processed_df['tags'] = processed_df['Course Name'] + ' ' + processed_df['Course Description'] + ' ' + processed_df['Skills']
    processed_df = processed_df[['Course Name', 'tags', 'Course Rating', 'Course URL']]
    
    # Clean text columns
    processed_df['Course Name'] = processed_df['Course Name'].astype(str)
    processed_df['tags'] = processed_df['tags'].astype(str).apply(clean_text)
    
    # Remove stopwords
    stop_words = load_stopwords()
    
    # More efficient stopword removal
    def remove_stopwords(text):
        return ' '.join([word for word in text.split() if word not in stop_words])
    
    processed_df['cleaned_tags'] = processed_df['tags'].apply(remove_stopwords)
    
    # Only use lemmatization if specifically requested (it's slow)
    nlp = get_nlp()
    if nlp:
        # Use batch processing for better performance
        docs = list(nlp.pipe(processed_df['cleaned_tags'], batch_size=50, disable=['ner', 'parser']))
        processed_df['lemmatized_tags'] = [' '.join([token.lemma_ for token in doc]) for doc in docs]
    else:
        # Simple stemming fallback if spaCy is not available
        def simple_stem(text):
            # Very simple stemmer - just removes common endings
            words = []
            for word in text.split():
                if word.endswith('ing'):
                    word = word[:-3]
                elif word.endswith('ed'):
                    word = word[:-2]
                elif word.endswith('s') and not word.endswith('ss'):
                    word = word[:-1]
                words.append(word)
            return ' '.join(words)
        
        processed_df['lemmatized_tags'] = processed_df['cleaned_tags'].apply(simple_stem)
    
    # Rename column for consistency
    processed_df.rename(columns={'Course Name': 'course_name'}, inplace=True)
    
    return processed_df

@st.cache_resource
def create_vectorizer_and_model(df):
    # Create TF-IDF vectorizer with fewer features for better performance
    tfidf = TfidfVectorizer(max_features=2000)
    vectors = tfidf.fit_transform(df['lemmatized_tags'])
    
    # Create KNN model
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    knn_model.fit(vectors)
    
    return tfidf, vectors, knn_model

def preprocess_input(course_query, tfidf_vectorizer):
    # Clean and preprocess user input
    course_query = clean_text(course_query)
    
    # Remove stopwords
    stop_words = load_stopwords()
    course_query = ' '.join([word for word in course_query.split() if word not in stop_words])
    
    # Lemmatize if available
    nlp = get_nlp()
    if nlp:
        course_query_processed = " ".join([token.lemma_ for token in nlp(course_query)])
    else:
        # Simple stemming fallback
        words = []
        for word in course_query.split():
            if word.endswith('ing'):
                word = word[:-3]
            elif word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('s') and not word.endswith('ss'):
                word = word[:-1]
            words.append(word)
        course_query_processed = ' '.join(words)
    
    # Transform to vector
    course_vector = tfidf_vectorizer.transform([course_query_processed])
    
    return course_vector

def recommend_courses(course_query, df, tfidf_vectorizer, vectors, knn_model, n_neighbors=5, use_knn=True):
    course_vector = preprocess_input(course_query, tfidf_vectorizer)
    if use_knn:
        distances, indices = knn_model.kneighbors(course_vector, n_neighbors=n_neighbors)
        return [
            {
                'course_name': df['course_name'].iloc[idx],
                'rating': df['Course Rating'].iloc[idx],
                'similarity': 1 - distances[0][i],
                'url': df['Course URL'].iloc[idx]
            }
            for i, idx in enumerate(indices[0])
        ]
    else:
        similarity_scores = cosine_similarity(course_vector, vectors)[0]
        top_indices = similarity_scores.argsort()[::-1][:n_neighbors]
        return [
            {
                'course_name': df['course_name'].iloc[idx],
                'rating': df['Course Rating'].iloc[idx],
                'similarity': similarity_scores[idx],
                'url': df['Course URL'].iloc[idx]
            }
            for idx in top_indices
        ]
    
    return recommended_courses

def calculate_metrics(recommendations):
    # Calculate metrics more efficiently using numpy
    ratings = np.array([rec['rating'] for rec in recommendations])
    similarities = np.array([rec['similarity'] for rec in recommendations])
    
    return ratings.mean(), similarities.mean(), np.mean(ratings > 4.5) * 100

def create_ratings_chart(recommendations):
    data = [(rec['course_name'][:20] + "..." if len(rec['course_name']) > 20 else rec['course_name'],
             rec['rating'], rec['similarity']) for rec in recommendations]
    courses, ratings, similarities = zip(*data)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=courses, y=ratings, name='Course Rating', marker_color='royalblue', opacity=0.7))
    fig.add_trace(go.Scatter(x=courses, y=[sim * 5 for sim in similarities], name='Similarity Score (scaled)',
                              mode='markers', marker=dict(size=12, color='firebrick', symbol='diamond')))
    fig.update_layout(title='Ratings and Similarity Scores', xaxis_title='Course', yaxis_title='Rating (0-5)',
                      barmode='group', template='plotly_white', height=400)
    return fig

def main():
    st.markdown('<h1 class="main-header">📚 Advanced Course Recommendation System</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 18px;">Welcome to the Course Recommendation System! This application helps you find the best courses based on your interests using AI-powered algorithms.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a debug mode toggle in sidebar
    with st.sidebar:
        debug_mode = st.checkbox("Debug Mode", value=False)
    
    # Load and preprocess data with progress indication
    with st.spinner("Loading and preparing data..."):
        df = load_data()
        
        if debug_mode:
            st.write("DataFrame shape:", df.shape)
            st.write("DataFrame columns:", df.columns.tolist())
        
        # Load processed DataFrame or create it if not in cache
        df_processed = preprocess_data(df)
        
        if debug_mode:
            st.write("Processed DataFrame shape:", df_processed.shape)
            st.write("Processed DataFrame columns:", df_processed.columns.tolist())
        
        # Create vectorizer and model
        tfidf_vectorizer, vectors, knn_model = create_vectorizer_and_model(df_processed)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 style="text-align: center; color: #1E88E5;">⚙️ Settings</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        recommendation_method = st.radio(
            "Recommendation Method",
            ["KNN-based", "Content-based (TF-IDF Similarity)"],
            index=0
        )
        
        num_recommendations = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
        
        if recommendation_method == "KNN-based":
            n_neighbors = st.slider(
                "Number of Neighbors (K)",
                min_value=3,
                max_value=15,
                value=5,
                step=2
            )
        else:
            n_neighbors = num_recommendations
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #004D40;">📊 Dataset Info</h3>', unsafe_allow_html=True)
        st.write(f"Total Courses: {df.shape[0]}")
        st.write(f"Average Rating: {df['Course Rating'].mean():.2f}/5.0")
        
        # Simple stat instead of full histogram for better performance
        st.write(f"Rating Range: {df['Course Rating'].min():.1f} - {df['Course Rating'].max():.1f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">🔍 Find Courses</h2>', unsafe_allow_html=True)
        
        # Course search
        course_query = st.text_input(
            "Enter a course topic, name or keywords:",
            placeholder="e.g., Machine Learning, Data Science, Python"
        )
        
        search_button = st.button("🔍 Find Recommendations", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">ℹ️ How it works</h2>', unsafe_allow_html=True)
        st.markdown("""
        1. **Enter keywords** related to courses you're interested in
        2. Choose between **KNN** or **Content-based** recommendation
        3. Get personalized course recommendations ranked by relevance and rating
        4. Click on course names to view more details
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show recommendations when search button is clicked
    if search_button and course_query:
        try:
            with st.spinner(f"Finding best courses matching '{course_query}'..."):
                # Get recommendations
                recommendations = recommend_courses(
                    course_query,
                    df_processed,
                    tfidf_vectorizer,
                    vectors,
                    knn_model,
                    n_neighbors=n_neighbors,
                    use_knn=(recommendation_method == "KNN-based")
                )
                
                # Display recommendations
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h2 class="sub-header">🎓 Top {len(recommendations)} Recommendations</h2>', unsafe_allow_html=True)
                
                # Calculate metrics
                avg_rating, avg_similarity, high_rated_percentage = calculate_metrics(recommendations)
                
                # Display metrics
                st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{avg_rating:.2f}</div>
                    <div class="metric-label">Average Rating</div>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{avg_similarity:.2f}</div>
                    <div class="metric-label">Average Similarity</div>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{high_rated_percentage:.1f}%</div>
                    <div class="metric-label">High-Rated Courses</div>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display chart
                st.plotly_chart(create_ratings_chart(recommendations), use_container_width=True)
                
                # Display recommendations more efficiently
                for i, rec in enumerate(recommendations):
                    st.markdown(f"""
                        <div class="recommendation-card">
                        <div class="course-title">
                            {i+1}. <a href="{rec['url']}" target="_blank" style="text-decoration: none; color: #0D47A1;">{rec['course_name']}</a>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                        <span class="course-rating">⭐ Rating: {rec['rating']:.2f}/5.0</span>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if debug_mode:
                # Add a toggle for debug mode
                debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
                if debug_mode:
                    st.write("DataFrame shape:", df.shape)
                    st.write("Processed DataFrame shape:", df_processed.shape)
                st.exception(e)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; background-color: rgba(255, 255, 255, 0.7); border-radius: 5px;">
        <p>📚 Course Recommendation System - Powered by Machine Learning Algorithms</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()