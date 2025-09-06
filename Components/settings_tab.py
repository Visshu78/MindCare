import streamlit as st
import random
from datetime import datetime

def settings_tab():
    st.header("⚙️ Settings & Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Account Settings")
        st.info(f"Logged in as: **{st.session_state.username}**")
        
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
        
        st.subheader("Notification Settings")
        reminder = st.checkbox("Enable daily reminders", value=True)
        if reminder:
            reminder_time = st.time_input("Reminder time", datetime.now().time())
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
    
    with col2:
        st.subheader("Mental Health Resources")
        
        resources = [
            {
                "name": "KIRAN Helpline",
                "number": "1800-599-0019",
                "desc": "Govt. of India 24x7 mental health rehabilitation helpline (multi-language)"
            },
            {
                "name": "AASRA",
                "number": "+91-9820466726",
                "desc": "24x7 suicide prevention & crisis support NGO"
            },
            {
                "name": "Vandrevala Foundation Helpline",
                "number": "+91-9999666555",
                "desc": "24x7 free mental health support across India"
            },
            {
                "name": "Snehi Helpline",
                "number": "+91-9582208181",
                "desc": "Emotional support and suicide prevention helpline"
            },
            {
                "name": "Emergency Services",
                "number": "100 / 108",
                "desc": "Police (100) or Ambulance (108) for immediate emergencies"
            }
        ]

        
        for resource in resources:
            with st.expander(f"{resource['name']}: {resource['number']}"):
                st.write(resource['desc'])
        
        st.subheader("Self-Care Ideas")
        self_care = [
            "Take a walk in nature 🌳",
            "Practice deep breathing for 5 minutes 🧘",
            "Listen to your favorite music 🎵",
            "Write down three things you're grateful for 📝",
            "Call a friend or loved one 📞",
            "Try a guided meditation 🪷"
        ]
        
        st.info("💡 **Today's self-care suggestion:**")
        st.success(random.choice(self_care))