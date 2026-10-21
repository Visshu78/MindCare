import re
import streamlit as st
from database import get_conn, init_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_valid_email(email: str) -> bool:
    """Return True if email matches a standard email pattern."""
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email.strip()) is not None


def email_exists(email: str) -> bool:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE email = ?', (email.strip().lower(),))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def username_exists(username: str) -> bool:
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ?', (username.strip(),))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def create_user(email: str, username: str, password: str) -> bool:
    """Insert new user. Returns True on success, False if email/username already taken."""
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute(
            'INSERT INTO users (email, username, password) VALUES (?, ?, ?)',
            (email.strip().lower(), username.strip(), password)
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def verify_user(identifier: str, password: str):
    """Return (user_id, username) tuple on success, or None. Matches email or username."""
    conn = get_conn()
    c = conn.cursor()
    ident = identifier.strip()
    c.execute(
        'SELECT id, username FROM users WHERE (LOWER(email) = ? OR LOWER(username) = ?) AND password = ?',
        (ident.lower(), ident.lower(), password)
    )
    row = c.fetchone()
    conn.close()
    return row  # (id, username) or None


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def check_authentication() -> bool:
    for key, default in [('user_id', None), ('username', None),
                         ('register', False), ('register_step', 1),
                         ('pending_email', ''), ('pending_password', '')]:
        if key not in st.session_state:
            st.session_state[key] = default
    return st.session_state.user_id is not None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def login_section():
    st.title("🧠 MindCare – Mental Health Journal")

    if st.session_state.register:
        _register_flow()
    else:
        _login_flow()

    # Footer
    st.markdown("---")
    st.subheader("Mental Health Resources")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Crisis Help**: National Suicide Prevention Lifeline: 988")
    with col2:
        st.info("**Support**: Crisis Text Line: Text HOME to 741741")
    with col3:
        st.info("**Therapy**: BetterHelp, Talkspace, or local therapists")


# ---------------------------------------------------------------------------
# Registration – two-step flow
# ---------------------------------------------------------------------------

def _register_flow():
    st.subheader("Create Account")

    if st.session_state.register_step == 1:
        _register_step1()
    else:
        _register_step2()


def _register_step1():
    """Step 1 – collect email + password."""
    st.markdown("**Step 1 of 2 — Enter your email and password**")

    with st.form("register_step1_form"):
        email = st.text_input("Email address", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Next →")

    if submitted:
        if not email.strip():
            st.error("Please enter an email address.")
        elif not is_valid_email(email):
            st.error("⚠️ That doesn't look like a valid email. Please check and try again.")
        elif email_exists(email):
            st.error("An account with this email already exists. Please log in.")
        elif not password:
            st.error("Please enter a password.")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            # Store temporarily and move to step 2
            st.session_state.pending_email = email.strip().lower()
            st.session_state.pending_password = password
            st.session_state.register_step = 2
            st.rerun()

    st.markdown("---")
    if st.button("← Back to Login"):
        st.session_state.register = False
        st.session_state.register_step = 1
        st.rerun()


def _register_step2():
    """Step 2 – choose a username."""
    st.markdown("**Step 2 of 2 — Choose a username**")
    st.info(f"Account email: `{st.session_state.pending_email}`")

    with st.form("register_step2_form"):
        username = st.text_input("Username", placeholder="e.g. mindful_warrior")
        submitted = st.form_submit_button("Create Account ✓")

    if submitted:
        if not username.strip():
            st.error("Please choose a username.")
        elif len(username.strip()) < 3:
            st.error("Username must be at least 3 characters.")
        elif username_exists(username):
            st.error("That username is already taken. Please choose another.")
        else:
            success = create_user(
                st.session_state.pending_email,
                username.strip(),
                st.session_state.pending_password,
            )
            if success:
                st.success("🎉 Account created! Please log in.")
                # Reset registration state
                st.session_state.register = False
                st.session_state.register_step = 1
                st.session_state.pending_email = ''
                st.session_state.pending_password = ''
                st.rerun()
            else:
                st.error("Something went wrong. Please try again.")

    st.markdown("---")
    if st.button("← Go back"):
        st.session_state.register_step = 1
        st.rerun()


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def _login_flow():
    st.subheader("Login to Your Journal")

    with st.form("login_form"):
        identifier = st.text_input("Username or Email", placeholder="Enter username or email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        if not identifier.strip():
            st.error("Please enter your username or email.")
        elif not password:
            st.error("Please enter your password.")
        else:
            row = verify_user(identifier, password)
            if row:
                st.session_state.user_id = row[0]
                st.session_state.username = row[1]
                st.rerun()
            else:
                st.error("❌ Invalid username/email or password.")

    st.markdown("---")
    if st.button("Create New Account"):
        st.session_state.register = True
        st.session_state.register_step = 1
        st.rerun()