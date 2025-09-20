import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re

# Ensure the same stopwords are used as in the notebook for consistency
additional_stopwords = {'the', 'of', 'and', 'in', 'a', 'to', 'for', 'on', 'with', 'from', 'as', 'by', 'at', 'an', 'is', 'that', 'this', 'are', 'be', 'was', 'have', 'it', 'its', 'or', 'new', 'study', 'report', 'case', 'review', 'paper', 'research', 'covid', 'sars', 'cov', '2', 'virus', 'disease'}

st.set_page_config(layout="wide")
st.title("CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research papers")

@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv('metadata.csv')
    except FileNotFoundError:
        st.error("Error: metadata.csv not found. Please ensure the file is in the same directory as app.py.")
        st.stop()

    # Data Cleaning and Preparation (replicated from notebook)
    missing_percentages = df.isnull().sum() / len(df) * 100
    drop_columns = missing_percentages[missing_percentages > 70].index
    df_cleaned = df.drop(columns=drop_columns)
    df_cleaned['title'] = df_cleaned['title'].fillna('')
    df_cleaned['abstract'] = df_cleaned['abstract'].fillna('')
    df_cleaned.dropna(subset=['publish_time'], inplace=True)
    df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
    df_cleaned.dropna(subset=['publish_time'], inplace=True)
    df_cleaned['publish_year'] = df_cleaned['publish_time'].dt.year
    df_cleaned['abstract_word_count'] = df_cleaned['abstract'].apply(lambda x: len(str(x).split()))
    df_cleaned['journal'] = df_cleaned['journal'].str.lower().str.strip()
    df_cleaned['source_x'] = df_cleaned['source_x'].str.lower().str.strip()
    return df_cleaned

df_cleaned = load_and_process_data()

# Sidebar filters
st.sidebar.header("Filter Options")

# Year Range Slider
if not df_cleaned.empty:
    min_year = int(df_cleaned['publish_year'].min())
    max_year = int(df_cleaned['publish_year'].max())
    year_range = st.sidebar.slider("Select Publication Year Range", min_year, max_year, (min_year, max_year))
    df_filtered = df_cleaned[(df_cleaned['publish_year'] >= year_range[0]) & (df_cleaned['publish_year'] <= year_range[1])]
else:
    st.warning("No data available for filtering.")
    df_filtered = df_cleaned

# Display sample data
st.header("Sample Data")
st.dataframe(df_filtered.head())

if not df_filtered.empty:
    # Publications over time
    st.header("Publications Over Time")
    year_counts = df_filtered['publish_year'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', ax=ax1)
    ax1.set_title('Number of Publications Over Time')
    ax1.set_xlabel('Publication Year')
    ax1.set_ylabel('Number of Papers')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    # Top Publishing Journals
    st.header("Top Publishing Journals")
    journal_counts = df_filtered['journal'].value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=journal_counts.values, y=journal_counts.index, palette='viridis', ax=ax2)
    ax2.set_title('Top 10 Publishing Journals')
    ax2.set_xlabel('Number of Papers')
    ax2.set_ylabel('Journal')
    st.pyplot(fig2)

    # Word Cloud of Paper Titles
    st.header("Word Cloud of Paper Titles")
    all_titles = " ".join(df_filtered['title'].dropna().astype(str))
    if all_titles:
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=additional_stopwords).generate(all_titles)
        fig3, ax3 = plt.subplots(figsize=(15, 8))
        ax3.imshow(wordcloud, interpolation='bilinear')
        ax3.axis('off')
        ax3.set_title('Word Cloud of Paper Titles')
        st.pyplot(fig3)
    else:
        st.write("No titles available to generate a word cloud.")

    # Distribution of Paper Counts by Source
    st.header("Distribution of Paper Counts by Source")
    source_counts = df_filtered['source_x'].value_counts().head(10)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=source_counts.values, y=source_counts.index, palette='magma', ax=ax4)
    ax4.set_title('Top 10 Paper Sources')
    ax4.set_xlabel('Number of Papers')
    ax4.set_ylabel('Source')
    st.pyplot(fig4)
else:
    st.warning("No data to display after filtering.")
