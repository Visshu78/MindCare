import streamlit as st
from datetime import datetime
from database import get_user_entries
from utils import create_pdf

def history_tab():
    st.header(" Journal History")
    
    entries = get_user_entries(st.session_state.user_id)
    
    if not entries:
        st.info("No journal entries yet. Start by writing your first entry!")
    else:
        # Search and filter options
        col1, col2 = st.columns([2, 1])
        with col1:
            search_text = st.text_input("Search entries")
        with col2:
            sentiment_filter = st.selectbox("Filter by sentiment", 
                                           ["All", "POSITIVE", "NEGATIVE", "NEUTRAL", "CRISIS"])
        
        # Export button
        if st.button("Export to PDF"):
            pdf_data = create_pdf(entries, st.session_state.username)
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=f"mental_health_journal_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        
        # Show filtered entries
        filtered_entries = entries
        if search_text:
            filtered_entries = [e for e in entries if search_text.lower() in e[1].lower()]
        if sentiment_filter != "All":
            filtered_entries = [e for e in filtered_entries if e[3] == sentiment_filter]
        
        if not filtered_entries:
            st.info("No entries match your filters.")
        else:
            for i, (date, entry_text, emotion, sentiment, confidence) in enumerate(filtered_entries):
                # Format the date for display
                display_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y")
                
                # Create expanders for each entry
                with st.expander(f"{display_date} - {emotion.capitalize()} ({sentiment})"):
                    st.write(entry_text)
                    sentiment_icon = "😊" if sentiment == "POSITIVE" else "😔" if sentiment == "NEGATIVE" else "🚨" if sentiment == "CRISIS" else "😐"
                    st.caption(f"{sentiment_icon} Emotion: {emotion} | Sentiment: {sentiment} ({(confidence*100):.1f}% confidence)")