import streamlit as st
import sqlite3
import hashlib
import secrets
import json
import os
import re
from datetime import datetime, timedelta

DB = "users.db"
TOKEN_FILE = ".session_token"


# ------------------- DATABASE -------------------
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        expires_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


# ------------------- PASSWORD & EMAIL -------------------
def hash_password(password, salt):
    return hashlib.sha256((password + salt).encode()).hexdigest()


def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least 1 capital letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least 1 number"
    if not re.search(r'[@!#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least 1 symbol (@, !, etc.)"
    return True, ""


def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


# ------------------- USER MANAGEMENT -------------------
def create_user(username, email, password):
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    if not validate_email(email):
        return False, "Invalid email address"
    
    is_valid, error_msg = validate_password(password)
    if not is_valid:
        return False, error_msg

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    salt = secrets.token_hex(16)
    pwd_hash = hash_password(password, salt)

    try:
        cur.execute(
            "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)",
            (username, email, pwd_hash, salt)
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return False, "Username already exists"
        elif 'email' in str(e):
            return False, "Email already exists"
        return False, "Username or email already exists"
    finally:
        conn.close()


def verify_user(username, password):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash, salt FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    user_id, stored_hash, salt = row
    if stored_hash == hash_password(password, salt):
        return user_id
    return None


def get_username(user_id):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


# ------------------- SESSION MANAGEMENT -------------------
def create_session(user_id, remember):
    token = secrets.token_urlsafe(32)
    expires = datetime.now() + (timedelta(days=7) if remember else timedelta(days=1))

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires.isoformat())
    )
    conn.commit()
    conn.close()

    if remember:
        save_token_to_file(token)

    return token


def get_session():
    token = get_token_from_file()
    if not token:
        return None

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,))
    row = cur.fetchone()
    conn.close()

    if not row:
        clear_token_file()
        return None

    user_id, expires = row
    expires = datetime.fromisoformat(expires)

    if datetime.now() > expires:
        delete_session(token)
        return None

    return user_id


def delete_session(token):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()
    
    clear_token_file()


def cleanup_expired_sessions():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE datetime(expires_at) < datetime('now')")
    conn.commit()
    conn.close()


def save_token_to_file(token):
    try:
        with open(TOKEN_FILE, 'w') as f:
            json.dump({'token': token}, f)
        if os.name != 'nt':
            os.chmod(TOKEN_FILE, 0o600)
    except Exception as e:
        print(f"Error saving token: {e}")


def get_token_from_file():
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'r') as f:
                data = json.load(f)
                return data.get('token')
    except:
        pass
    return None


def clear_token_file():
    try:
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
    except Exception as e:
        print(f"Error clearing token: {e}")


def check_authentication():
    if st.session_state.get("authenticated", False):
        return True, st.session_state.get("username")
    
    user_id = get_session()
    if user_id:
        username = get_username(user_id)
        if username:
            st.session_state.update({
                "authenticated": True,
                "user_id": user_id,
                "username": username
            })
            return True, username
    
    return False, None


# ------------------- UI -------------------
def login_page():
    init_db()
    cleanup_expired_sessions()
    st.title("📖 My Learning AI")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    # ---------- LOGIN ----------
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            remember = st.checkbox("Remember me for 7 days")
            submit = st.form_submit_button("Login")

            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    user_id = verify_user(username, password)
                    if user_id:
                        create_session(user_id, remember)
                        st.session_state.update({
                            "authenticated": True,
                            "user_id": user_id,
                            "username": username,
                            "_rerun_flag": True
                        })
                        st.success(f"Welcome back, {username}!")
                    else:
                        st.error("Invalid username or password")

    # ---------- SIGNUP ----------
    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            st.caption("Password must be at least 8 characters with 1 capital letter, 1 number, and 1 symbol (@, !, etc.)")
            signup = st.form_submit_button("Create Account")

            if signup:
                if not new_username or not new_email or not new_password:
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = create_user(new_username, new_email, new_password)
                    if success:
                        st.success(message)
                        st.info("Please login with your new account")
                    else:
                        st.error(message)

    # ---------- SAFE RERUN ----------
    if st.session_state.get("_rerun_flag"):
        st.session_state["_rerun_flag"] = False
        st.experimental_rerun()


def logout():
    token = get_token_from_file()
    if token:
        delete_session(token)
    st.session_state.clear()
    st.experimental_rerun()
