import streamlit as st
import pandas as pd
import plotly.express as px
from analysis_engine import process_data  # Assuming you wrap your logic in a function

# Page Configuration
st.set_page_config(page_title="Laundry Insights Dashboard", layout="wide")

st.title("Indonesia Laundry Business Intelligence")
st.markdown("Analyzing market trends, customer sentiment, and geographic hotspots.")

# --- SIDEBAR & DATA LOADING ---
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload Scraped CSV", type="csv")

if uploaded_file:
    # Run your backend processing
    with st.spinner('Processing NLP and Topic Modeling...'):
        df, topics = process_data(uploaded_file)
    
    # --- METRICS ROW ---
    col1, col2 = st.columns(2)
    col1.metric("Total Businesses", len(df))
    col2.metric("Avg Rating", round(df['review_rating'].mean(), 2))

    # --- TABS FOR DIFFERENT VIEWS ---
    tab1, tab2, tab3 = st.tabs(["Geographic Map", "Market Leaderboard", "Topic Analysis"])

    with tab1:
        st.subheader("High-Performance Locations")
        # Plotly map is interactive (zoom/hover)
        fig_map = px.scatter_mapbox(
            df.head(50), 
            lat="latitude", 
            lon="longitude", 
            size="performance_score",
            color="review_rating",
            hover_name="title",
            hover_data=["review_count", "complete_address"],
            zoom=12, 
            height=600,
            color_continuous_scale=px.colors.cyclical.IceFire
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        st.subheader("Top Performing Businesses")
        st.dataframe(
            df[['title', 'review_rating', 'review_count', 'performance_score']]
            .sort_values(by="performance_score", ascending=False)
            .head(20),
            use_container_width=True
        )

    with tab3:
        st.subheader("What are customers saying?")
        cols = st.columns(len(topics))
        for i, topic_words in enumerate(topics):
            with cols[i]:
                st.info(f"**Theme {i+1}**")
                st.write(topic_words)

else:
    st.info("Please upload the `dataset_raw_from_webscraping.csv` file in the sidebar to begin.")