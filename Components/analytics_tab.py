import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from database import get_user_entries

def analytics_tab():
    st.header(" Mood Analytics")
    
    entries = get_user_entries(st.session_state.user_id)
    
    if not entries:
        st.info("No journal entries yet. Start by writing your first entry!")
    else:
        # Prepare data for visualization
        df = pd.DataFrame(entries, columns=['date', 'entry_text', 'emotion', 'sentiment', 'confidence'])
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Map sentiment to numerical values for plotting
        sentiment_map = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
        df['sentiment_value'] = df['sentiment'].map(sentiment_map)
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Mood trend chart
            st.subheader("Mood Trend Over Time")
            fig = px.line(
                df, 
                x='date', 
                y='sentiment_value',
                title='Your Emotional Journey',
                labels={'sentiment_value': 'Mood Score', 'date': 'Date'}
            )
            fig.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2],
                    ticktext=['Negative', 'Neutral', 'Positive']
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Emotion distribution
            st.subheader("Emotion Distribution")
            emotion_counts = df['emotion'].value_counts()
            fig2 = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title='Your Emotional Patterns'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Statistics
        st.subheader("Your Journaling Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entries", len(entries))
        with col2:
            positive_count = len(df[df['sentiment'] == 'POSITIVE'])
            st.metric("Positive Days", positive_count)
        with col3:
            streak = 0
            current_date = datetime.now().date()
            for i in range(len(df)):
                if (current_date - timedelta(days=i)).strftime("%Y-%m-%d") in df['date'].dt.strftime("%Y-%m-%d").values:
                    streak += 1
                else:
                    break
            st.metric("Current Streak", f"{streak} days")
        with col4:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg. Confidence", f"{avg_confidence*100:.1f}%")