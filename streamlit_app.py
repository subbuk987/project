import streamlit as st
import requests
import tempfile
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Floating Waste Detection",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        color: white;
    }
    .stats-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .category-header {
        color: #0d6efd;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #0d6efd;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def create_tracking_visualization(tracking_stats):
    # Prepare data for visualization
    data = []
    for category, stats in tracking_stats.items():
        for obj in stats['objects']:
            data.append({
                'Category': category,
                'Object ID': f"#{obj['id']}",
                'Duration': obj['duration']
            })
    
    if data:
        df = pd.DataFrame(data)
        
        # Create pie chart for category distribution
        fig_pie = px.pie(
            df['Category'].value_counts().reset_index(),
            values='Category',
            names='index',
            title='Waste Distribution by Category',
            hole=0.4
        )
        
        # Create bar chart for object counts
        fig_counts = px.bar(
            df['Category'].value_counts().reset_index(),
            x='index',
            y='Category',
            title='Objects Detected by Category',
            labels={'index': 'Category', 'Category': 'Count'},
            color='index'
        )
        
        # Create box plot for duration distribution
        fig_duration = px.box(
            df,
            x='Category',
            y='Duration',
            title='Detection Duration Distribution by Category',
            color='Category'
        )
        
        # Create timeline visualization
        fig_timeline = px.scatter(
            df,
            x='Duration',
            y='Category',
            color='Category',
            title='Object Detection Timeline',
            labels={'Duration': 'Frame Number'},
            size=[10] * len(df)
        )
        
        return fig_pie, fig_counts, fig_duration, fig_timeline
    
    return None, None, None, None

def display_metric_card(title, value, description=""):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{title}</div>
            <div class="metric-description">{description}</div>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.title("🌊 Floating Waste Detection System")
    
    # Add description
    st.markdown("""
    This system detects and tracks floating waste in video footage using YOLOv8 and ByteTrack.
    Upload a video to analyze the presence of different types of waste materials.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save the uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Process the video
        status_text.text("Processing video...")
        files = {'video': open(tfile.name, 'rb')}
        response = requests.post('http://localhost:5000/process_video', files=files)
        
        if response.status_code == 200:
            result = response.json()
            tracking_stats = result['tracking_stats']
            
            # Update progress
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            # Display processed video
            st.subheader("Processed Video")
            st.video(f"http://localhost:5000{result['video_path']}")
            
            # Dashboard Section
            st.markdown("---")
            st.header("📊 Waste Detection Dashboard")
            
            # Summary metrics
            total_objects = sum(stats['count'] for stats in tracking_stats.values())
            
            # Display summary metrics in columns
            metric_cols = st.columns(4)
            with metric_cols[0]:
                display_metric_card("Total Objects", total_objects, "Total waste items detected")
            
            for idx, (category, stats) in enumerate(tracking_stats.items(), 1):
                with metric_cols[idx]:
                    display_metric_card(
                        f"{category.title()}", 
                        stats['count'],
                        f"Unique {category} items"
                    )
            
            # Detailed Statistics Section
            st.subheader("📈 Detailed Analysis")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Visualizations", "📋 Detailed Statistics"])
            
            with tab1:
                # Get visualizations
                fig_pie, fig_counts, fig_duration, fig_timeline = create_tracking_visualization(tracking_stats)
                
                if fig_pie and fig_counts and fig_duration and fig_timeline:
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.plotly_chart(fig_pie, use_container_width=True)
                        st.plotly_chart(fig_duration, use_container_width=True)
                    
                    with viz_col2:
                        st.plotly_chart(fig_counts, use_container_width=True)
                        st.plotly_chart(fig_timeline, use_container_width=True)
            
            with tab2:
                # Display detailed statistics for each category
                for category, stats in tracking_stats.items():
                    with st.expander(f"{category.title()} Category Details", expanded=True):
                        st.markdown(f"""
                        <div class="stats-container">
                            <h3 class="category-header">{category.title()} Statistics</h3>
                            <p><strong>Total Objects:</strong> {stats['count']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if stats['count'] > 0:
                            df = pd.DataFrame(stats['objects'])
                            df.columns = ['Object ID', 'Duration (frames)']
                            st.dataframe(df, use_container_width=True)
                            
                            # Add average duration
                            avg_duration = df['Duration (frames)'].mean()
                            st.info(f"Average detection duration: {avg_duration:.2f} frames")
            
        else:
            st.error("Error processing the video. Please try again.")
        
        # Clean up
        os.unlink(tfile.name)

if __name__ == "__main__":
    main()