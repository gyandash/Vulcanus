import streamlit as st
import os
from dotenv import load_dotenv
from utils.auth import (
    authenticate_user_neo4j,
    logout_user,
    register_user,
    is_admin_user,
    get_pending_users,
    approve_user,
    validate_password_policy
)
from components.ui_styles import apply_custom_styles
from streamlit_cookies_controller import CookieController

# Load environment variables from .env file
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=dotenv_path)

# Initialize CookieManager
cookies = CookieController()

# --- Page Configuration ---
st.set_page_config(
    page_title="Vulcanus AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles
apply_custom_styles()

# --- Feature Card Rendering Function ---
def render_feature_cards():
    st.markdown("---")
    st.subheader("Key Features")
    st.markdown("""
        <style>
            .feature-cards-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 30px;
                margin-bottom: 50px;
            }
            .feature-card {
                background: rgba(40, 40, 65, 0.45);
                border: 1px solid rgba(160, 130, 255, 0.25);
                border-radius: 15px;
                padding: 25px 20px;
                text-align: center;
                box-shadow: 0 0 12px rgba(100, 80, 255, 0.15);
                transition: all 0.25s ease;
            }
            .feature-card:hover {
                transform: translateY(-4px);
                background: rgba(60, 60, 95, 0.6);
                box-shadow: 0 0 25px rgba(160, 130, 255, 0.25);
            }
            .feature-card .card-icon { font-size: 2.5em; margin-bottom: 10px; }
            .feature-card .card-title { font-size: 1.2em; color: #CBB9FF; margin-bottom: 8px; }
            .feature-card .card-description { font-size: 0.9em; color: #BFC4E8; line-height: 1.4; }
        </style>
        <div class="feature-cards-container">
    """, unsafe_allow_html=True)

    features = [
        {"icon": "💻", "title": "AI Code Generation", "description": "Generate, convert, refactor, and optimize code across languages."},
        {"icon": "📊", "title": "Data Analysis & Charting", "description": "Upload data, query AI for insights, and visualize with interactive charts."},
        {"icon": "📄", "title": "Document Processing", "description": "Upload documents to extract text and query their content with AI."},
        {"icon": "🗺️", "title": "Project Flow Mapping", "description": "Visualize project, data, or process flows using AI-generated diagrams."},
        {"icon": "☁️", "title": "Cloud Code Conversion", "description": "Seamlessly convert and update cloud-specific code between platforms/services."},
        {"icon": "🖼️", "title": "AI Wireframe UI Generator", "description": "Generate UI wireframes from natural language descriptions using custom markup."},
        {"icon": "🧠", "title": "Context-Aware Memory Agent", "description": "Remembers your preferences, past tasks, and project context. Offers personalized suggestions and auto-completes based on history."},
        {"icon": "🤝", "title": "Multi-Agent Collaboration", "description": "Simulate collaboration between AI agents to design complete systems and solutions."}
    ]

    for feature in features:
        st.markdown(f"""
            <div class="feature-card">
                <div class="card-icon">{feature['icon']}</div>
                <h3 class="card-title">{feature['title']}</h3>
                <p class="card-description">{feature['description']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Authentication Logic ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = cookies.get("authenticated") == "true"
    st.session_state.username = cookies.get("username") if st.session_state.authenticated else ""
    st.session_state.show_register_form = False
    st.session_state.show_admin_panel = False

if not st.session_state.authenticated:
    st.sidebar.title("🔮 Login / Register")
    st.sidebar.markdown("---")

    # Display Hero Section + Features before login
    st.markdown("""
        <style>
        /* Hero Section Styling */
        .hero-section {
            text-align: center;
            background: linear-gradient(145deg, rgba(25,25,45,0.8), rgba(15,15,25,0.95));
            border: 1px solid rgba(160,130,255,0.15);
            border-radius: 16px;
            padding: 40px 25px;
            margin-top: 20px;
            margin-bottom: 40px;
            box-shadow: 0 0 35px rgba(140,120,255,0.12);
            transition: all 0.3s ease-in-out;
        }
        .hero-section:hover {
            box-shadow: 0 0 45px rgba(160,130,255,0.25);
            transform: translateY(-2px);
        }
        .hero-title {
            font-size: 2.6rem;
            color: #CBB9FF;
            text-shadow: 0px 0px 15px rgba(160,130,255,0.3);
            margin-bottom: 10px;
            font-weight: 700;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            color: #AEB6FF;
            margin-bottom: 18px;
            font-weight: 400;
        }
        .hero-desc {
            font-size: 1rem;
            color: #D1D1E9;
            line-height: 1.7;
            max-width: 750px;
            margin: 0 auto;
        }
        hr.custom-divider {
            margin: 25px auto;
            border: none;
            height: 1px;
            width: 60%;
            background: linear-gradient(to right, rgba(120,90,255,0.2), rgba(255,255,255,0.15), rgba(120,90,255,0.2));
        }
        </style>

        <div class="hero-section">
            <div class="hero-title">✨ Vulcanus AI</div>
            <div class="hero-subtitle">Your Unified Intelligent Developer Workspace</div>
            <hr class="custom-divider">
            <p class="hero-desc">
                Welcome to <b>Vulcanus AI</b> — where intelligence meets automation.<br>
                Streamline your development workflow with powerful AI agents that can code, analyze data,
                design interfaces, and collaborate autonomously — all in one place.
            </p>
        </div>
    """, unsafe_allow_html=True)

    render_feature_cards()

    if st.session_state.show_register_form:
        with st.sidebar.form(key="register_form"):
            st.subheader("Register New User")
            new_username = st.text_input("New Username", key="register_username_input_form")
            new_password = st.text_input("New Password", type="password", key="register_password_input_form")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_input_form")
            register_submit_button = st.form_submit_button("Register")

            if new_password:
                errors = validate_password_policy(new_password)
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    st.success("Password meets policy requirements!")

            if register_submit_button:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif not new_username.strip() or not new_password.strip():
                    st.error("Username and password cannot be empty.")
                else:
                    success, message = register_user(new_username, new_password)
                    if success:
                        st.success(message)
                        st.session_state.show_register_form = False
                        st.rerun()
                    else:
                        st.error(message)
        st.sidebar.button("Back to Login", key="back_to_login_button", on_click=lambda: st.session_state.update(show_register_form=False))
    else:
        with st.sidebar.form(key="login_form"):
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_username_input_form")
            password = st.text_input("Password", type="password", key="login_password_input_form")
            login_submit_button = st.form_submit_button("Login")

            if login_submit_button:
                cookies.set("authenticated", "false")
                cookies.set("username", "")
                success, message = authenticate_user_neo4j(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    cookies.set("authenticated", "true")
                    cookies.set("username", username)
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error(message)

        st.sidebar.button("Register New User", key="show_register_form_button", on_click=lambda: st.session_state.update(show_register_form=True))
    st.stop()

# --- Post-login section ---
st.sidebar.success(f"Welcome, {st.session_state.username}!")

if is_admin_user(st.session_state.username):
    st.sidebar.markdown("---")
    st.sidebar.subheader("👑 Admin Panel")
    if st.sidebar.button("Manage User Approvals", key="manage_user_approvals_button"):
        st.session_state.show_admin_panel = not st.session_state.show_admin_panel
        st.rerun()

    if st.session_state.get("show_admin_panel"):
        st.subheader("User Approval Management")
        pending_users = get_pending_users()
        if pending_users:
            st.info("The following users are pending approval:")
            for user in pending_users:
                col_user, col_approve = st.columns([0.7, 0.3])
                with col_user:
                    st.write(f"**Username:** {user['username']}")
                    st.write(f"Registered: {user['created_at'].strftime('%Y-%m-%d %H:%M')}")
                with col_approve:
                    if st.button(f"Approve {user['username']}", key=f"approve_user_{user['username']}"):
                        if approve_user(user['username']):
                            st.success(f"User '{user['username']}' approved successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to approve user '{user['username']}'.")
                st.markdown("---")
        else:
            st.info("No users currently pending approval.")
    st.markdown("---")

if st.sidebar.button("Logout", key="logout_button"):
    logout_user()
    st.session_state.authenticated = False
    st.session_state.username = ""
    cookies.set("authenticated", "false")
    cookies.set("username", "")
    st.rerun()

# --- MAIN CONTENT ---
st.markdown("---")
st.markdown(
    """
    ## 🚀 Explore Vulcanus AI Features
    - **💻 Code Generator:** Generate, convert, and refactor code across languages.
    - **📊 Data Analysis:** Upload CSV/XLSX and visualize insights.
    - **📄 Document Processor:** Upload and query document content.
    - **🗺️ Project Flow Mapper:** Generate conceptual or data flow diagrams.
    - **☁️ Cloud Code Converter:** Transform cloud-specific code across services.
    - **🖼️ AI Wireframe Generator:** Convert UI ideas into wireframes.
    - **🧠 Context-Aware Memory Agent:** Remembers your preferences and offers personalized assistance.
    - **🤝 Multi-Agent Collaboration:** Simulate teamwork between AI agents to design complete systems.
    """
)

render_feature_cards()