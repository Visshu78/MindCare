import os
import sys
import streamlit as st
from database import get_conn
from auth import (
    is_valid_email, email_exists, username_exists,
    create_user, verify_user, check_authentication, login_section
)

# Ensure project root is in sys.path so model_experiments is accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@st.cache_resource(show_spinner="Loading MindCare Deep Learning Model (DistilBERT + BiGRU + Attention)...")
def load_model():
    """Load the trained MindCare Proposed Deep Learning Model (DistilBERT + BiGRU + Attention)."""
    try:
        from model_experiments.predictor import MindCarePredictor
        checkpoint_path = os.path.join(project_root, "model_experiments", "saved_models", "proposed_best.pt")
        predictor = MindCarePredictor(checkpoint_path=checkpoint_path)
        return predictor
    except Exception as e:
        st.warning(f"Custom model fallback note ({e}). Falling back to standard pipeline.")
        try:
            from transformers import pipeline
            return pipeline("sentiment-analysis")
        except Exception:
            return None


def sidebar_content():
    """Display sidebar content."""
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
    """Create a PDF from journal entries."""
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

        title = Paragraph(f"MindCare Journal – {username}", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))

        export_date = Paragraph(
            f"Exported on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']
        )
        story.append(export_date)
        story.append(Spacer(1, 12))

        for date, entry_text, emotion, sentiment, confidence in entries:
            date_header = Paragraph(
                f"<b>{datetime.strptime(date, '%Y-%m-%d').strftime('%B %d, %Y')}</b>",
                styles['Heading2']
            )
            story.append(date_header)

            entry_para = Paragraph(entry_text, styles['Normal'])
            story.append(entry_para)

            sentiment_info = Paragraph(
                f"<i>Emotion: {emotion.capitalize()} | Sentiment: {sentiment} "
                f"({confidence * 100:.1f}% confidence)</i>",
                styles['Italic']
            )
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
