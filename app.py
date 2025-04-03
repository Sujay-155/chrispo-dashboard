import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image, ImageFilter, ImageEnhance
import os
import random
import io
import base64
import glob
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from datetime import datetime

# Download NLTK resources once using caching
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")
        return False

# Create necessary directories for image storage
@st.cache_resource
def create_directories():
    for day in range(1, 6):
        os.makedirs(f"temp_uploads/day_{day}", exist_ok=True)
    return True

# Generate sample sports images if they don't exist (using colored placeholders)
@st.cache_resource
def generate_sample_images():
    # Define sports
    sports = ['Basketball', 'Football', 'Tennis', 'Swimming', 'Athletics', 
              'Volleyball', 'Badminton', 'Cricket', 'Chess', 'Table Tennis']
    
    # Create images for each day and sport
    img_count = 0
    for day in range(1, 6):
        day_dir = f"sample_images/day_{day}"
        os.makedirs(day_dir, exist_ok=True)
        
        # Select random 3-4 sports for each day
        num_sports = random.randint(3, 4)
        selected_sports = random.sample(sports, num_sports)
        
        for sport in selected_sports:
            # Don't regenerate if image already exists
            if os.path.exists(f"{day_dir}/{sport}.jpg"):
                continue
                
            # Create colored image with text
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            img = Image.new('RGB', (400, 300), color=color)
            draw = Image.new('RGBA', img.size, (255, 255, 255, 0))
            
            # Draw text
            from PIL import ImageDraw
            d = ImageDraw.Draw(draw)
            text = f"{sport}\nDay {day}"
            text_position = (img.width // 2, img.height // 2)
            d.text(text_position, text, fill="white", anchor="mm")
            
            # Combine the images
            img = Image.alpha_composite(img.convert('RGBA'), draw)
            
            # Save image
            img_path = f"{day_dir}/{sport}.jpg"
            img.convert('RGB').save(img_path)
            img_count += 1
    
    return img_count

# ===============================
# PART 1: DATASET GENERATION
# ===============================
@st.cache_data
def generate_dataset():
    """
    Generate a synthetic dataset for CHRISPO '25 sports event
    """
    np.random.seed(42)  # For reproducibility
    
    # Define constants
    NUM_PARTICIPANTS = 300
    NUM_DAYS = 5
    SPORTS = ['Basketball', 'Football', 'Tennis', 'Swimming', 'Athletics', 
              'Volleyball', 'Badminton', 'Cricket', 'Chess', 'Table Tennis']
    
    COLLEGES = ['Engineering College', 'Arts & Science College', 'Medical College', 
                'Law College', 'Business School', 'Technology Institute', 
                'Physical Education College', 'Science Institute']
    
    STATES = ['Kerala', 'Tamil Nadu', 'Karnataka', 'Maharashtra', 'Delhi', 
              'West Bengal', 'Gujarat', 'Telangana', 'Andhra Pradesh', 'Punjab']
    
    # Generate participant basic info
    participant_ids = range(1, NUM_PARTICIPANTS + 1)
    names = [f"Participant_{i}" for i in participant_ids]
    ages = np.random.randint(18, 30, NUM_PARTICIPANTS)
    genders = np.random.choice(['Male', 'Female'], NUM_PARTICIPANTS)
    colleges = np.random.choice(COLLEGES, NUM_PARTICIPANTS)
    states = np.random.choice(STATES, NUM_PARTICIPANTS)
    
    # Generate participation data
    data = []
    
    for participant_id, name, age, gender, college, state in zip(
        participant_ids, names, ages, genders, colleges, states):
        
        # Each participant participates in 1-3 sports
        num_sports = np.random.randint(1, 4)
        selected_sports = np.random.choice(SPORTS, num_sports, replace=False)
        
        for sport in selected_sports:
            # Each sport participation happens on 1 specific day
            day = np.random.randint(1, NUM_DAYS + 1)
            
            # Generate performance score (1-10)
            performance = np.random.randint(1, 11)
            
            # Generate feedback based on performance
            if performance >= 8:
                sentiment = "positive"
                feedback_templates = [
                    f"Really enjoyed the {sport} event! Well organized.",
                    f"Great experience at {sport}. Would participate again!",
                    f"The {sport} competition was fantastic and challenging.",
                    f"Excellent facilities for {sport}. Very professional setup."
                ]
            elif performance >= 5:
                sentiment = "neutral"
                feedback_templates = [
                    f"The {sport} event was okay. Some improvements needed.",
                    f"{sport} competition was average, but enjoyed participating.",
                    f"Decent organization of the {sport} event.",
                    f"The {sport} facilities were acceptable."
                ]
            else:
                sentiment = "negative"
                feedback_templates = [
                    f"The {sport} event could be better organized.",
                    f"Faced some issues during the {sport} competition.",
                    f"The {sport} event needs improvement in timing and management.",
                    f"Not satisfied with how the {sport} competition was conducted."
                ]
            
            feedback = np.random.choice(feedback_templates)
            
            data.append({
                'ParticipantID': participant_id,
                'Name': name,
                'Age': age,
                'Gender': gender,
                'College': college,
                'State': state,
                'Sport': sport,
                'Day': day,
                'Performance': performance,
                'Feedback': feedback,
                'Sentiment': sentiment
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

# ===============================
# PART 2: STREAMLIT DASHBOARD
# ===============================
def create_dashboard():
    # Setup page configuration
    st.set_page_config(layout="wide", page_title="CHRISPO '25")
    
    # Download NLTK resources and create directories
    with st.spinner("Setting up resources..."):
        download_nltk_resources()
        create_directories()
    
    # Generate or load data
    with st.spinner("Preparing dataset..."):
        try:
            data = generate_dataset()
            st.sidebar.success("Dataset generated successfully")
        except Exception as e:
            st.error(f"Error generating dataset: {e}")
            return
    
    st.title("CHRISPO '25 Sports Event Dashboard")
    st.markdown("### Interactive Dashboard for CHRISPO '25 Sports Event Analysis")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Participation Analysis", "Text Analysis", "Image Processing"])
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Prepare filter options
    sports = sorted(data['Sport'].unique())
    days = sorted(data['Day'].unique())
    colleges = sorted(data['College'].unique())
    states = sorted(data['State'].unique())
    
    selected_sports = st.sidebar.multiselect("Select Sports", sports)
    selected_days = st.sidebar.multiselect("Select Days", days)
    selected_colleges = st.sidebar.multiselect("Select Colleges", colleges)
    selected_states = st.sidebar.multiselect("Select States", states)
    
    # Apply filters
    filtered_data = data.copy()
    if selected_sports:
        filtered_data = filtered_data[filtered_data['Sport'].isin(selected_sports)]
    if selected_days:
        filtered_data = filtered_data[filtered_data['Day'].isin(selected_days)]
    if selected_colleges:
        filtered_data = filtered_data[filtered_data['College'].isin(selected_colleges)]
    if selected_states:
        filtered_data = filtered_data[filtered_data['State'].isin(selected_states)]
    
    # If no data after filtering
    if filtered_data.empty:
        st.warning("No data available with selected filters. Please adjust your selection.")
        return
    
    # ==========================
    # TAB 1: PARTICIPATION ANALYSIS
    # ==========================
    with tab1:
        st.header("Participation Analysis")
        
        # Layout for charts
        col1, col2 = st.columns(2)
        
        # Chart 1: Sports Participation
        with col1:
            sports_counts = filtered_data['Sport'].value_counts().reset_index()
            sports_counts.columns = ['Sport', 'Count']
            fig1 = px.bar(sports_counts, x='Sport', y='Count', 
                        title='Participation by Sports',
                        color='Sport')
            st.plotly_chart(fig1, use_container_width=True)
        
        # Chart 2: Day-wise Participation
        with col2:
            day_counts = filtered_data['Day'].value_counts().sort_index().reset_index()
            day_counts.columns = ['Day', 'Count']
            fig2 = px.line(day_counts, x='Day', y='Count', 
                         title='Participation by Day',
                         markers=True)
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        # Chart 3: College-wise Participation
        with col3:
            college_counts = filtered_data['College'].value_counts().reset_index()
            college_counts.columns = ['College', 'Count']
            fig3 = px.pie(college_counts, names='College', values='Count',
                        title='Participation by College')
            st.plotly_chart(fig3, use_container_width=True)
        
        # Chart 4: State-wise Participation
        with col4:
            state_counts = filtered_data['State'].value_counts().reset_index()
            state_counts.columns = ['State', 'Count']
            fig4 = px.scatter(filtered_data, x='Age', y='Performance', 
                            color='State', size='Performance',
                            title='Performance by Age and State')
            st.plotly_chart(fig4, use_container_width=True)
        
        # Chart 5: Performance Heatmap
        heatmap_data = filtered_data.pivot_table(
            values='Performance', index='Sport', columns='Day', aggfunc='mean')
        fig5 = px.imshow(heatmap_data, 
                        title='Average Performance Heatmap (Sport vs Day)',
                        labels=dict(x="Day", y="Sport", color="Avg Performance"))
        st.plotly_chart(fig5, use_container_width=True)
    
    # ==========================
    # TAB 2: TEXT ANALYSIS
    # ==========================
    with tab2:
        st.header("Feedback Text Analysis")
        
        # Word cloud for selected sport
        st.subheader("Word Cloud by Sport")
        selected_sport_wc = st.selectbox("Select Sport for Word Cloud", sports)
        
        sport_feedback = data[data['Sport'] == selected_sport_wc]['Feedback'].str.cat(sep=' ')
        
        # Generate word cloud
        if sport_feedback:
            stop_words = set(stopwords.words('english'))
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white', 
                                stopwords=stop_words,
                                max_words=100).generate(sport_feedback)
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No feedback data available for the selected sport.")
        
        # Sentiment Analysis
        st.subheader("Feedback Sentiment Analysis by Sport")
        
        sentiment_counts = data.groupby(['Sport', 'Sentiment']).size().reset_index()
        sentiment_counts.columns = ['Sport', 'Sentiment', 'Count']
        
        fig_sentiment = px.bar(sentiment_counts, x='Sport', y='Count', color='Sentiment',
                           title='Feedback Sentiment Analysis by Sport',
                           barmode='group')
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Top common words
        st.subheader("Most Common Words in Feedback")
        all_feedback = " ".join(data['Feedback'])
        words = all_feedback.lower().split()
        word_counts = pd.Series(words).value_counts().head(20)
        
        fig_words = px.bar(x=word_counts.index, y=word_counts.values, 
                         labels={'x': 'Word', 'y': 'Frequency'},
                         title='Most Common Words in Feedback')
        st.plotly_chart(fig_words, use_container_width=True)
    
    # ==========================
    # TAB 3: IMAGE PROCESSING
    # ==========================
    with tab3:
        st.header("Image Processing Module")
        
        # Generate sample images if we don't have any
        if not os.path.exists("sample_images"):
            with st.spinner("Generating sample sports images..."):
                os.makedirs("sample_images", exist_ok=True)
                img_count = generate_sample_images()
                st.success(f"Generated {img_count} sample images for the gallery")
        
        # Image Gallery Section
        st.subheader("Day-wise Image Gallery")
        view_day = st.selectbox("Select Day to View Gallery:", range(1, 6), key="gallery_day")
        
        # Look for images in either sample_images or temp_uploads
        image_files = []
        
        # Check sample images directory first
        sample_dir = f"sample_images/day_{view_day}"
        if os.path.exists(sample_dir):
            image_files.extend(glob.glob(f"{sample_dir}/*.jpg"))
            image_files.extend(glob.glob(f"{sample_dir}/*.jpeg"))
            image_files.extend(glob.glob(f"{sample_dir}/*.png"))
        
        # Check user uploads directory
        temp_dir = f"temp_uploads/day_{view_day}"
        if os.path.exists(temp_dir):
            image_files.extend(glob.glob(f"{temp_dir}/*.jpg"))
            image_files.extend(glob.glob(f"{temp_dir}/*.jpeg"))
            image_files.extend(glob.glob(f"{temp_dir}/*.png"))
        
        if image_files:
            st.markdown(f"### Sports Events - Day {view_day}")
            
            # Create columns for gallery display
            cols = st.columns(2)
            
            for i, image_file in enumerate(image_files):
                try:
                    with cols[i % 2]:
                        img = Image.open(image_file)
                        sport_name = os.path.basename(image_file).split('.')[0]
                        st.image(img, caption=f"{sport_name} - Day {view_day}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image {image_file}: {str(e)}")
        else:
            st.info(f"No images found for Day {view_day}.")
        
        # Add option to upload additional images
        st.markdown("---")
        st.subheader("Upload More Gallery Images")
        
        upload_day = st.selectbox("Select day to upload additional images for:", range(1, 6), key="upload_day")
        uploaded_gallery_images = st.file_uploader(
            "Upload more images for the gallery",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="additional_gallery_images"
        )
        
        if uploaded_gallery_images:
            # Create directory if it doesn't exist
            os.makedirs(f"temp_uploads/day_{upload_day}", exist_ok=True)
            
            for img_file in uploaded_gallery_images:
                # Save the uploaded image
                img_path = f"temp_uploads/day_{upload_day}/{img_file.name}"
                with open(img_path, "wb") as f:
                    f.write(img_file.getbuffer())
                st.success(f"Added {img_file.name} to Day {upload_day} gallery!")
        
        # Custom Image Processing Section
        st.markdown("---")
        st.subheader("Custom Image Processing")
        
        # Two options: Select from gallery or upload new
        processing_source = st.radio(
            "Select image source for processing:",
            ["Upload New Image", "Use Image from Gallery"]
        )
        
        image_to_process = None
        
        if processing_source == "Upload New Image":
            uploaded_file = st.file_uploader(
                "Upload an image for processing",
                type=["jpg", "jpeg", "png"],
                key="processing_image"
            )
            if uploaded_file:
                image_to_process = Image.open(uploaded_file)
        else:
            # Select from gallery
            select_day = st.selectbox("Select day:", range(1, 6), key="select_process_day")
            
            # Combine files from both directories
            gallery_files = []
            
            # Check sample images first
            if os.path.exists(f"sample_images/day_{select_day}"):
                gallery_files.extend(glob.glob(f"sample_images/day_{select_day}/*.jpg"))
                gallery_files.extend(glob.glob(f"sample_images/day_{select_day}/*.jpeg"))
                gallery_files.extend(glob.glob(f"sample_images/day_{select_day}/*.png"))
            
            # Check user uploads
            if os.path.exists(f"temp_uploads/day_{select_day}"):
                gallery_files.extend(glob.glob(f"temp_uploads/day_{select_day}/*.jpg"))
                gallery_files.extend(glob.glob(f"temp_uploads/day_{select_day}/*.jpeg"))
                gallery_files.extend(glob.glob(f"temp_uploads/day_{select_day}/*.png"))
            
            if gallery_files:
                file_names = [os.path.basename(f) for f in gallery_files]
                selected_file = st.selectbox("Select image from gallery:", file_names)
                
                if selected_file:
                    # Look in both directories
                    file_path = None
                    
                    # Check sample_images first
                    temp_path = f"sample_images/day_{select_day}/{selected_file}"
                    if os.path.exists(temp_path):
                        file_path = temp_path
                    
                    # If not found, check temp_uploads
                    if file_path is None:
                        temp_path = f"temp_uploads/day_{select_day}/{selected_file}"
                        if os.path.exists(temp_path):
                            file_path = temp_path
                    
                    if file_path:
                        image_to_process = Image.open(file_path)
            else:
                st.warning(f"No images found for Day {select_day}")
        
        # Process the image if we have one
        if image_to_process:
            # Effect selection
            effect = st.selectbox(
                "Select an effect to apply",
                ["None", "Blur", "Sharpen", "Contour", "Enhance", "Grayscale", "Sepia", "Adjust Colors"]
            )
            
            # Parameter sliders based on selected effect
            params = {}
            if effect == "Blur":
                params['radius'] = st.slider("Blur Radius", 1, 20, 5)
            elif effect == "Enhance":
                params['factor'] = st.slider("Enhancement Factor", 0.0, 5.0, 1.5)
            elif effect == "Adjust Colors":
                params['brightness'] = st.slider("Brightness", 0.0, 2.0, 1.0)
                params['contrast'] = st.slider("Contrast", 0.0, 2.0, 1.0)
                params['saturation'] = st.slider("Saturation", 0.0, 2.0, 1.0)
            
            # Display original and processed images side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image_to_process, use_column_width=True)
            
            with col2:
                st.subheader("Processed Image")
                
                # Apply the selected effect
                try:
                    if effect == "Blur":
                        processed_image = image_to_process.filter(ImageFilter.GaussianBlur(params['radius']))
                    elif effect == "Sharpen":
                        processed_image = image_to_process.filter(ImageFilter.SHARPEN)
                    elif effect == "Contour":
                        processed_image = image_to_process.filter(ImageFilter.CONTOUR)
                    elif effect == "Enhance":
                        enhancer = ImageEnhance.Contrast(image_to_process)
                        processed_image = enhancer.enhance(params.get('factor', 2.0))
                    elif effect == "Grayscale":
                        processed_image = image_to_process.convert('L')
                    elif effect == "Sepia":
                        # Create sepia effect
                        grayscale = image_to_process.convert('L')
                        sepia = Image.new('RGB', image_to_process.size)
                        for x in range(image_to_process.width):
                            for y in range(image_to_process.height):
                                gray_value = grayscale.getpixel((x, y))
                                sepia.putpixel((x, y), (min(gray_value + 100, 255), 
                                                       min(gray_value + 50, 255), 
                                                       max(gray_value - 20, 0)))
                        processed_image = sepia
                    elif effect == "Adjust Colors":
                        # Apply multiple enhancements
                        temp_image = image_to_process
                        
                        enhancer = ImageEnhance.Brightness(temp_image)
                        temp_image = enhancer.enhance(params['brightness'])
                        
                        enhancer = ImageEnhance.Contrast(temp_image)
                        temp_image = enhancer.enhance(params['contrast'])
                        
                        enhancer = ImageEnhance.Color(temp_image)
                        processed_image = enhancer.enhance(params['saturation'])
                    else:
                        processed_image = image_to_process
                    
                    # Display processed image
                    st.image(processed_image, use_column_width=True)
                    
                    # Option to download the processed image
                    buf = io.BytesIO()
                    processed_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download processed image",
                        data=byte_im,
                        file_name=f"processed_{effect.lower()}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

# Add footer with app info
def add_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center">
            <p>CHRISPO '25 Dashboard | Created with Streamlit</p>
            <p>© 2025 | [GitHub Repository](https://github.com/your-username/chrispo-dashboard)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main function
def main():
    # Initialize app
    create_dashboard()
    add_footer()

if __name__ == "__main__":
    main()