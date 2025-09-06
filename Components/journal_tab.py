import streamlit as st
from datetime import datetime
import random
from database import add_user_entry

def journal_tab(sentiment_pipeline):
    st.header("Today's Journal Entry")
    
    # Date selector with today as default
    selected_date = st.date_input("Select Date", datetime.now())
    
    # Text area for journal entry
    journal_entry = st.text_area(
        "How are you feeling today?",
        height=200,
        placeholder="Write about your thoughts, feelings, and experiences..."
    )
    
    # Mood slider
    mood_rating = st.slider("Rate your overall mood today", 1, 10, 5)
    st.write(f"Mood rating: {mood_rating}/10")
    
    # Analyze sentiment when user submits an entry
    if st.button("Save Entry", type="primary"):
        if journal_entry.strip():
            # Perform sentiment analysis
            crisis_keywords = ["suicide", "kill myself", "die", "dying", "end it all", "want to die"]
            if any(word in journal_entry.lower() for word in crisis_keywords):
                emotion = "CRISIS"
                sentiment = "CRISIS"
                confidence = 1.0
            else:
                if sentiment_pipeline:
                    result = sentiment_pipeline(journal_entry)[0]
                    emotion = result['label']
                    sentiment = map_emotion_to_sentiment(emotion)
                    confidence = result['score']
                else:
                    # Fallback if model is not available
                    emotion = "NEUTRAL"
                    sentiment = "NEUTRAL"
                    confidence = 0.5
            
            # Format date for storage
            formatted_date = selected_date.strftime("%Y-%m-%d")
            
            # Save to database
            add_user_entry(st.session_state.user_id, formatted_date, journal_entry, 
                          emotion, sentiment, confidence)
            
            # Show feedback based on sentiment
            feedback_messages = {
                "POSITIVE": [
                    "Keep shining! Your positivity is inspiring 🌟",
                    "You're doing amazing! Keep up the great work 💪"
                ],
                "NEGATIVE": [
                    "It's okay to have difficult days. Be gentle with yourself 🌸",
                    "Consider taking a short break - maybe listen to calming music 🎶"
                ],
                "NEUTRAL": [
                    "Balance is good. Notice what brings you peace today 🍃",
                    "Every day is a new opportunity for growth 🌱"
                ],
                "CRISIS": [
                    "⚠️ It seems you're going through a very difficult time. "
                    "Please reach out to a mental health professional or crisis helpline immediately. "
                    "You are not alone. 💙"
                ]
            }
            
            feedback = random.choice(feedback_messages.get(sentiment, ["Thank you for sharing."]))
            emoji = "✅" if sentiment == "POSITIVE" else "🤗" if sentiment == "NEGATIVE" else "🚨" if sentiment == "CRISIS" else "😊"
            
            st.success("Entry saved successfully!")
            st.markdown(f"### {emoji} {feedback}")
            
            # Show sentiment results
            sentiment_color = "green" if sentiment == "POSITIVE" else "red" if sentiment == "NEGATIVE" else "orange" if sentiment == "CRISIS" else "blue"
            st.markdown(f"**Detected emotion:** {emotion.capitalize()}")
            st.markdown(f"**Overall sentiment:** :{sentiment_color}[{sentiment}] ({(confidence*100):.1f}% confidence)")
        else:
            st.warning("Please write something before saving.")

# Emotion to sentiment mapping (duplicated here for this component)
def map_emotion_to_sentiment(emotion_label):
    emotion_mapping = {
        'joy': 'POSITIVE', 'happy': 'POSITIVE', 'optimism': 'POSITIVE', 'love': 'POSITIVE',
        'anger': 'NEGATIVE', 'annoyance': 'NEGATIVE', 'disgust': 'NEGATIVE', 
        'sadness': 'NEGATIVE', 'fear': 'NEGATIVE', 'pessimism': 'NEGATIVE',
        'surprise': 'NEUTRAL', 'neutral': 'NEUTRAL', 'confusion': 'NEUTRAL'
    }
    return emotion_mapping.get(emotion_label.lower(), 'NEUTRAL')