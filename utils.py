import streamlit as st
import sqlite3
from database import init_db

def create_user(username, password):
    conn = sqlite3.connect('journal.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                 (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('journal.db')
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ? AND password = ?', 
             (username, password))
    user = c.fetchone()
    conn.close()
    return user[0] if user else None

def check_authentication():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'register' not in st.session_state:
        st.session_state.register = False
        
    return st.session_state.user_id is not None

def login_section():
    st.title("MindCare - Mental Health Journal")
    
    if st.session_state.register:
        st.subheader("Create Account")
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if new_password == confirm_password:
                    if create_user(new_username, new_password):
                        st.success("Account created successfully! Please login.")
                        st.session_state.register = False
                    else:
                        st.error("Username already exists.")
                else:
                    st.error("Passwords do not match.")
        
        if st.button("Back to Login"):
            st.session_state.register = False
    else:
        st.subheader("Login to Your Journal")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                user_id = verify_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        
        if st.button("Create New Account"):
            st.session_state.register = True
    
    # Footer with resources
    st.markdown("---")
    st.subheader("Mental Health Resources")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Crisis Help**: National Suicide Prevention Lifeline: 988")
    with col2:
        st.info("**Support**: Crisis Text Line: Text HOME to 741741")
    with col3:
        st.info("**Therapy**: BetterHelp, Talkspace, or local therapists")

def load_model():
    """Load sentiment analysis model"""
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis")
    except ImportError:
        st.warning("Transformers library not available. Sentiment analysis will be disabled.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def sidebar_content():
    """Display sidebar content"""
    with st.sidebar:
        st.markdown("### Welcome to MindCare!")
        st.markdown("Track your mental health journey with daily journal entries.")
        
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Quick Tips")
        st.markdown("• Write regularly for better insights")
        st.markdown("• Be honest about your feelings")
        st.markdown("• Review your patterns in Analytics")
        st.markdown("• Seek help if you need it")

def create_pdf(entries, username):
    """Create a PDF from journal entries"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        import io
        from datetime import datetime
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph(f"MindCare Journal - {username}", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Export date
        export_date = Paragraph(f"Exported on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal'])
        story.append(export_date)
        story.append(Spacer(1, 12))
        
        # Entries
        for date, entry_text, emotion, sentiment, confidence in entries:
            # Date header
            date_header = Paragraph(f"<b>{datetime.strptime(date, '%Y-%m-%d').strftime('%B %d, %Y')}</b>", styles['Heading2'])
            story.append(date_header)
            
            # Entry text
            entry_para = Paragraph(entry_text, styles['Normal'])
            story.append(entry_para)
            
            # Sentiment info
            sentiment_info = Paragraph(f"<i>Emotion: {emotion.capitalize()} | Sentiment: {sentiment} ({confidence*100:.1f}% confidence)</i>", styles['Italic'])
            story.append(sentiment_info)
            story.append(Spacer(1, 12))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        st.error("ReportLab library not available. PDF export is disabled.")
        return None
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None
