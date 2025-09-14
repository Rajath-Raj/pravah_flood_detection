import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌊 Pravah - Flood Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load the flood dataset with caching for better performance"""
    try:
        df = pd.read_csv('flood_dataset_classification.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset file 'flood_dataset_classification.csv' not found!")
        st.info("Please make sure the CSV file is in the same directory as this app.")
        return None

# Data preprocessing function
@st.cache_data
def preprocess_data(df):
    """Preprocess the data and create target variable"""
    # Drop unwanted columns if they exist
    columns_to_drop = ['Total Deaths', 'Total Affected', 'duration', 'Disaster Type']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)
    
    # Create target variable if it doesn't exist
    if 'Is_Severe_Flood' not in df.columns:
        rainfall_threshold = df['Rainfall'].quantile(0.75)
        df['Is_Severe_Flood'] = (df['Rainfall'] > rainfall_threshold).astype(int)
    
    return df

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 Pravah - Flood Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predicting Flood Events Using Machine Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Exploration", "📈 Visualizations", "🤖 Model Training", "🔮 Predictions"]
    )
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📊 Data Exploration":
        show_data_exploration(df)
    elif page == "📈 Visualizations":
        show_visualizations(df)
    elif page == "🤖 Model Training":
        show_model_training(df)
    elif page == "🔮 Predictions":
        show_predictions(df)

def show_home_page(df):
    """Display the home page with project overview"""
    st.markdown('<h2 class="sub-header">📋 Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Total Records",
            value=f"{len(df):,}",
            delta="Complete dataset"
        )
    
    with col2:
        st.metric(
            label="📋 Features",
            value=f"{len(df.columns)-1}",
            delta="Input variables"
        )
    
    with col3:
        flood_percentage = (df['Is_Severe_Flood'].sum() / len(df)) * 100
        st.metric(
            label="🌊 Severe Floods",
            value=f"{flood_percentage:.1f}%",
            delta="Of total events"
        )
    
    st.markdown("---")
    
    # Project description
    st.markdown("""
    ### 🎯 What is Pravah?
    **Pravah** (प्रवाह - meaning "flow" in Sanskrit) is a machine learning project designed to predict severe flood events 
    based on geographical and environmental characteristics.
    
    ### 🔬 How it works:
    1. **Data Collection**: Uses historical disaster data with features like rainfall, elevation, slope, etc.
    2. **Data Analysis**: Explores patterns and relationships in the data
    3. **Model Training**: Trains machine learning algorithms to recognize flood patterns
    4. **Prediction**: Makes predictions on new data to identify potential severe floods
    
    ### 📈 Current Status:
    - ✅ Data loaded and preprocessed
    - ✅ Exploratory data analysis complete
    - ✅ Target variable created (severe floods based on rainfall threshold)
    - ✅ Ready for model training and predictions
    """)
    
    # Quick data preview
    st.markdown('<h3 class="sub-header">👀 Quick Data Preview</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

def show_data_exploration(df):
    """Display data exploration page"""
    st.markdown('<h2 class="sub-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
    
    # Dataset information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Dataset Info")
        st.write(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.markdown("### 🔍 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count()
        })
        st.dataframe(dtype_df)
    
    with col2:
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(df.describe())
        
        st.markdown("### ❌ Missing Values")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values found!")
        else:
            st.dataframe(missing_data[missing_data > 0])
    
    # Target variable distribution
    st.markdown("### 🎯 Target Variable Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        df['Is_Severe_Flood'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Distribution of Severe vs Non-Severe Floods')
        ax.set_xlabel('Is Severe Flood')
        ax.set_ylabel('Count')
        ax.set_xticklabels(['Non-Severe', 'Severe'], rotation=0)
        st.pyplot(fig)
    
    with col2:
        value_counts = df['Is_Severe_Flood'].value_counts()
        fig = px.pie(
            values=value_counts.values,
            names=['Non-Severe', 'Severe'],
            title='Flood Event Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_visualizations(df):
    """Display data visualizations"""
    st.markdown('<h2 class="sub-header">📈 Data Visualizations</h2>', unsafe_allow_html=True)
    
    # Feature selection
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Is_Severe_Flood' in numeric_columns:
        numeric_columns.remove('Is_Severe_Flood')
    
    # Correlation heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = df[numeric_columns + ['Is_Severe_Flood']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)
    
    # Feature distributions
    st.markdown("### 📊 Feature Distributions")
    selected_features = st.multiselect(
        "Select features to visualize:",
        numeric_columns,
        default=numeric_columns[:3]
    )
    
    if selected_features:
        cols = st.columns(min(len(selected_features), 3))
        for i, feature in enumerate(selected_features):
            with cols[i % 3]:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[feature].hist(bins=30, ax=ax, alpha=0.7)
                ax.set_title(f'{feature} Distribution')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
    
    # Scatter plots
    st.markdown("### 🎯 Feature Relationships")
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("Select X-axis:", numeric_columns, index=0)
    
    with col2:
        y_axis = st.selectbox("Select Y-axis:", numeric_columns, index=1)
    
    if x_axis and y_axis:
        fig = px.scatter(
            df, x=x_axis, y=y_axis, 
            color='Is_Severe_Flood',
            color_discrete_map={0: 'blue', 1: 'red'},
            title=f'{x_axis} vs {y_axis} (Colored by Flood Severity)',
            labels={'Is_Severe_Flood': 'Severe Flood'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_training(df):
    """Display model training page"""
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    # Prepare data
    X = df.drop('Is_Severe_Flood', axis=1)
    y = df['Is_Severe_Flood']
    
    # Model parameters
    st.markdown("### ⚙️ Model Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    
    with col2:
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
    
    with col3:
        random_state = st.number_input("Random State", 0, 100, 42)
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_names = X.columns.tolist()
            
            # Display results
            st.success(f"✅ Model trained successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Accuracy", f"{accuracy:.3f}")
            
            with col2:
                st.metric("📊 Training Samples", len(X_train))
            
            with col3:
                st.metric("🧪 Test Samples", len(X_test))
            
            # Confusion Matrix
            st.markdown("### 🔍 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # Classification Report
            st.markdown("### 📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Feature Importance
            st.markdown("### 🌟 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df.head(10), 
                x='Importance', 
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Features'
            )
            st.plotly_chart(fig, use_container_width=True)

def show_predictions(df):
    """Display prediction interface"""
    st.markdown('<h2 class="sub-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)
    
    # Check if both model and scaler exist
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        st.warning("⚠️ Please train a model first in the 'Model Training' section!")
        st.info("👈 Go to the 'Model Training' page to train your model, then return here to make predictions.")
        return
    
    st.markdown("### 📝 Enter Feature Values")
    
    # Get feature names (excluding target)
    feature_names = st.session_state.feature_names
    
    # Create input fields
    col1, col2 = st.columns(2)
    input_data = {}
    
    for i, feature in enumerate(feature_names):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            # Get feature statistics for reasonable defaults
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=f"input_{feature}"
            )
    
    # Make prediction
    if st.button("🔮 Predict Flood Risk", type="primary"):
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        input_scaled = st.session_state.scaler.transform(input_df)
        
        # Make prediction
        prediction = st.session_state.model.predict(input_scaled)[0]
        prediction_proba = st.session_state.model.predict_proba(input_scaled)[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("🚨 HIGH FLOOD RISK")
                st.markdown("**Severe flood event predicted!**")
            else:
                st.success("✅ LOW FLOOD RISK")
                st.markdown("**Non-severe flood event predicted**")
        
        with col2:
            st.metric(
                "🎯 Confidence (Non-Severe)",
                f"{prediction_proba[0]:.3f}"
            )
        
        with col3:
            st.metric(
                "🌊 Confidence (Severe)",
                f"{prediction_proba[1]:.3f}"
            )
        
        # Probability chart
        fig = go.Figure(data=[
            go.Bar(name='Probability', x=['Non-Severe', 'Severe'], y=prediction_proba)
        ])
        fig.update_layout(title='Prediction Probabilities')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()