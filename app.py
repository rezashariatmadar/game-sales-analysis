# python -m streamlit run app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Page config
st.set_page_config(
    page_title="Game Sales Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved CSS that works with both light and dark themes
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid rgba(28, 131, 225, 0.8);
    }
    .prediction-box {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid rgba(28, 131, 225, 0.8);
    }
    .report-box {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid rgba(28, 131, 225, 0.8);
    }
    </style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title('🎮 Video Game Sales: Comprehensive Analysis')
st.markdown("""
This interactive dashboard provides a comprehensive analysis of video game sales data, 
including market trends, regional performance, and predictive insights.
""")

# --- Load Data and Models ---
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').dropna().astype(int)
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        
        # Debug: Print column names to help identify the issue
        st.write("Available columns:", df.columns.tolist())
        
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found. Please make sure it's in the correct directory.")
        return None

@st.cache_resource
def load_models():
    try:
        models = {
            'regression': joblib.load('regression_results/random_forest_model.joblib'),
            'naive_bayes': joblib.load('naive_bayes_results/naive_bayes_model.joblib'),
            'decision_tree': joblib.load('decision_tree_results/decision_tree_model.joblib')
        }
        scaler = joblib.load('regression_results/scaler.joblib')
        return models, scaler
    except FileNotFoundError:
        st.warning("Model files not found. Prediction features will be disabled.")
        return None, None

# Load data and models
df_cleaned = load_data('vgchartz_cleaned.csv')
models, scaler = load_models()

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Interactive Analysis", "📈 Prediction", "📑 Analysis Reports"])

# Check if data is loaded
if df_cleaned is not None:
    # Interactive Analysis Tab
    with tab1:
        # Sidebar Filters
        st.sidebar.header('📊 Analysis Filters')
        
        # Year Range Slider
        min_year, max_year = int(df_cleaned['release_year'].min()), int(df_cleaned['release_year'].max())
        year_range = st.sidebar.slider(
            'Select Release Year Range:',
            min_year, 
            max_year, 
            (2000, 2016),
            help="Filter games by their release year"
        )

        # Genre Selection
        genres = sorted(df_cleaned['genre'].unique())
        selected_genres = st.sidebar.multiselect(
            'Select Genres:', 
            genres, 
            default=['Action', 'Sports', 'Shooter'],
            help="Choose one or more game genres to analyze"
        )

        # Platform Selection - Check if 'platform' or 'console' is available
        platform_col = 'platform' if 'platform' in df_cleaned.columns else 'console'
        if platform_col in df_cleaned.columns:
            platforms = sorted(df_cleaned[platform_col].unique())
            selected_platforms = st.sidebar.multiselect(
                f'Select {platform_col.title()}s:', 
                platforms, 
                default=platforms[:5] if len(platforms) >= 5 else platforms,
                help=f"Choose one or more gaming {platform_col}s"
            )
        else:
            st.sidebar.warning(f"No platform/console column found in the data.")
            selected_platforms = []
            platforms = []

        # Publisher Selection
        publishers = sorted(df_cleaned['publisher'].unique())
        selected_publishers = st.sidebar.multiselect(
            'Select Publishers:', 
            publishers, 
            default=publishers[:5] if len(publishers) >= 5 else publishers,
            help="Choose one or more game publishers"
        )

        # Filter DataFrame
        filter_conditions = [
            (df_cleaned['release_year'] >= year_range[0]),
            (df_cleaned['release_year'] <= year_range[1]),
            (df_cleaned['genre'].isin(selected_genres)),
            (df_cleaned['publisher'].isin(selected_publishers))
        ]
        
        # Add platform filter if available
        if platform_col in df_cleaned.columns and selected_platforms:
            filter_conditions.append(df_cleaned[platform_col].isin(selected_platforms))
            
        # Apply all filters
        df_filtered = df_cleaned[np.logical_and.reduce(filter_conditions)]

        # Main Content
        st.header("📈 Market Overview")
        
        # Key Metrics in a more visually appealing format
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Games", len(df_filtered))
        with col2:
            st.metric("Total Sales (M)", f"{df_filtered['total_sales'].sum():.2f}")
        with col3:
            st.metric("Avg. Critic Score", f"{df_filtered['critic_score'].mean():.1f}")
        with col4:
            st.metric("Unique Publishers", df_filtered['publisher'].nunique())

        # Market Trends
        st.subheader("Market Trends Over Time")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales Trend
            yearly_sales = df_filtered.groupby('release_year')['total_sales'].sum().reset_index()
            fig_sales = px.line(
                yearly_sales, 
                x='release_year', 
                y='total_sales',
                title='Total Sales by Year',
                labels={'total_sales': 'Sales (Millions)', 'release_year': 'Year'}
            )
            fig_sales.update_layout(
                template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"
            )
            st.plotly_chart(fig_sales, use_container_width=True)

        with col2:
            # Genre Distribution
            genre_sales = df_filtered.groupby('genre')['total_sales'].sum().sort_values(ascending=True)
            fig_genre = px.bar(
                genre_sales, 
                orientation='h',
                title='Sales by Genre',
                labels={'value': 'Sales (Millions)', 'index': 'Genre'}
            )
            fig_genre.update_layout(
                template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"
            )
            st.plotly_chart(fig_genre, use_container_width=True)

        # Regional Analysis
        st.header("🌍 Regional Performance")
        col1, col2 = st.columns(2)

        with col1:
            # Regional Sales Distribution
            regional_sales = df_filtered[['na_sales', 'pal_sales', 'jp_sales', 'other_sales']].sum()
            fig_regional = px.pie(
                values=regional_sales.values,
                names=['North America', 'Europe', 'Japan', 'Other'],
                title='Regional Sales Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_regional.update_layout(
                template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"
            )
            st.plotly_chart(fig_regional, use_container_width=True)

        with col2:
            # Top Publishers by Region
            publisher_sales = df_filtered.groupby('publisher')[['na_sales', 'pal_sales', 'jp_sales']].sum()
            publisher_sales = publisher_sales.nlargest(10, 'na_sales')
            fig_publisher = px.bar(
                publisher_sales,
                title='Top 10 Publishers by Regional Sales',
                labels={'value': 'Sales (Millions)', 'variable': 'Region'},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_publisher.update_layout(
                template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"
            )
            st.plotly_chart(fig_publisher, use_container_width=True)

        # Platform Analysis
        if platform_col in df_cleaned.columns:
            st.header(f"🎮 {platform_col.title()} Performance")
            col1, col2 = st.columns(2)

            with col1:
                # Platform Sales
                platform_sales = df_filtered.groupby(platform_col)['total_sales'].sum().sort_values(ascending=True)
                fig_platform = px.bar(
                    platform_sales,
                    orientation='h',
                    title=f'Sales by {platform_col.title()}',
                    labels={'value': 'Sales (Millions)', 'index': platform_col.title()},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_platform.update_layout(
                    template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"
                )
                st.plotly_chart(fig_platform, use_container_width=True)

            with col2:
                # Platform-Genre Heatmap
                try:
                    platform_genre = pd.crosstab(df_filtered[platform_col], df_filtered['genre'])
                    fig_heatmap = px.imshow(
                        platform_genre,
                        title=f'{platform_col.title()}-Genre Distribution',
                        labels={'x': 'Genre', 'y': platform_col.title()},
                        color_continuous_scale='Viridis'
                    )
                    fig_heatmap.update_layout(
                        template="plotly_white" if st.get_option("theme.base") == "light" else "plotly_dark"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating heatmap: {str(e)}")

        # Data Table
        st.header("📋 Detailed Data View")
        st.dataframe(
            df_filtered.sort_values('total_sales', ascending=False).head(20),
            use_container_width=True
        )

        # Download Button
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name=f"game_sales_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # Prediction Tab
    with tab2:
        if models is not None:
            st.header("🎯 Sales Prediction")
            st.markdown("""
            Use the form below to predict potential sales for a new game based on its characteristics.
            """)
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    pred_genre = st.selectbox("Genre", genres)
                    pred_platform = st.selectbox("Platform", platforms if 'platforms' in locals() and len(platforms) > 0 else ["PS4", "Xbox One", "Switch"])
                    pred_publisher = st.selectbox("Publisher", publishers)
                    pred_year = st.number_input("Release Year", min_value=1980, max_value=2024, value=2024)
                
                with col2:
                    pred_critic_score = st.slider("Expected Critic Score", 0, 100, 75)
                    pred_na_sales = st.number_input("Expected NA Sales (M)", min_value=0.0, max_value=50.0, value=1.0)
                    pred_pal_sales = st.number_input("Expected PAL Sales (M)", min_value=0.0, max_value=50.0, value=1.0)
                    pred_jp_sales = st.number_input("Expected JP Sales (M)", min_value=0.0, max_value=50.0, value=1.0)
                
                submit_button = st.form_submit_button("Predict Sales")
                
                if submit_button:
                    try:
                        # Get the features that were used in training
                        # The create_models.py script used: critic_score, release_year, na_sales, jp_sales, pal_sales
                        
                        # Prepare input data with exact same features and order as training
                        input_data = pd.DataFrame({
                            'critic_score': [pred_critic_score],
                            'release_year': [pred_year], 
                            'na_sales': [pred_na_sales],
                            'jp_sales': [pred_jp_sales],
                            'pal_sales': [pred_pal_sales]
                        })
                        
                        # Debug information
                        st.write("Features used for prediction:", input_data.columns.tolist())
                        
                        # Scale numerical features
                        if scaler is not None:
                            input_data_scaled = scaler.transform(input_data)
                        
                        # Make predictions
                        regression_pred = models['regression'].predict(input_data_scaled)[0]
                        
                        # Display prediction using Streamlit components instead of HTML
                        st.success(f"### Predicted Total Sales: {regression_pred:.2f} million units")
                        st.info("Based on the Random Forest Regression model")
                        
                        # Show confidence metrics
                        st.subheader("Model Confidence")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("R-squared Score", "0.9523")
                        with col2:
                            st.metric("Mean Absolute Error", "0.0721")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.error("Debugging info: Make sure the features match those used in training (critic_score, release_year, na_sales, jp_sales, pal_sales)")
        else:
            st.info("Prediction models are not available. Please ensure model files are in the correct location.")

    # Analysis Reports Tab
    with tab3:
        st.header("📑 Analysis Reports")
        
        # Regression Analysis Report - Using Streamlit native components
        st.subheader("Regression Analysis")
        with st.expander("View Regression Analysis Report", expanded=True):
            st.markdown("#### Model Performance")
            st.markdown("- R-squared (R²): 0.9523 (95.23% variance explained)")
            st.markdown("- Mean Absolute Error: 0.0721 (Low prediction error)")
            
            st.markdown("#### Feature Importance")
            st.markdown("- Japan sales ratio: 56.58%")
            st.markdown("- North America ratio: 21.47%")
            st.markdown("- PAL region ratio: 20.46%")
            
            st.markdown("#### Key Insights")
            st.markdown("- Regional performance profiles are the strongest predictors")
            st.markdown("- Games with focused regional strategies show more predictable outcomes")
            st.markdown("- Critic scores and traditional factors have minimal impact")
        
        # Naive Bayes Report - Using Streamlit native components
        st.subheader("Naive Bayes Classification")
        with st.expander("View Naive Bayes Classification Report", expanded=True):
            st.markdown("#### Model Performance")
            st.markdown("- Gaussian Naive Bayes Accuracy: 84.40%")
            st.markdown("- Bernoulli Naive Bayes Accuracy: 55.10%")
            
            st.markdown("#### Key Findings")
            st.markdown("- Gaussian Naive Bayes performs significantly better for continuous features")
            st.markdown("- Model shows good performance in identifying successful games")
            st.markdown("- Feature distributions are well-captured by the Gaussian model")
        
        # Decision Tree Report - Using Streamlit native components
        st.subheader("Decision Tree Analysis")
        with st.expander("View Decision Tree Analysis Report", expanded=True):
            st.markdown("#### Model Performance")
            st.markdown("- Accuracy: 98.00%")
            st.markdown("- High precision in classification tasks")
            
            st.markdown("#### Key Insights")
            st.markdown("- Clear decision paths for game success prediction")
            st.markdown("- Strong feature importance hierarchy")
            st.markdown("- Effective in capturing non-linear relationships")
        
        # Clustering Reports - Using Streamlit native components
        st.subheader("Clustering Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("Hierarchical Clustering", expanded=True):
                st.markdown("#### Hierarchical Clustering")
                st.markdown("- Identified distinct market segments")
                st.markdown("- Clear regional patterns in clusters")
                st.markdown("- Genre-platform relationships revealed")
        
        with col2:
            with st.expander("K-means Clustering", expanded=True):
                st.markdown("#### K-means Clustering")
                st.markdown("- Optimal clusters: 5")
                st.markdown("- Clear separation of market segments")
                st.markdown("- Strong regional patterns identified")
        
        # Display relevant images if they exist
        st.subheader("Visual Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('regression_results/feature_importance.png'):
                st.image('regression_results/feature_importance.png', caption='Feature Importance')
            if os.path.exists('naive_bayes_results/roc_curves.png'):
                st.image('naive_bayes_results/roc_curves.png', caption='ROC Curves')
        
        with col2:
            if os.path.exists('decision_tree_results/decision_tree_visualization.png'):
                st.image('decision_tree_results/decision_tree_visualization.png', caption='Decision Tree Visualization')
            if os.path.exists('clustering_results/kmeans_pca_visualization.png'):
                st.image('clustering_results/kmeans_pca_visualization.png', caption='K-means Clustering')

else:
    st.error("Please ensure the data file is available in the correct location.") 