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