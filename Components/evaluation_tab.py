import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def load_metrics_data(dataset_key="emotion"):
    """Load benchmark metrics and confusion matrices."""
    filename = "reddit_summary_metrics.json" if dataset_key == "reddit" else "summary_metrics.json"
    comp_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.abspath(os.path.join(comp_dir, "..", "model_experiments", "results", filename)),
        os.path.abspath(os.path.join(comp_dir, "..", "..", "model_experiments", "results", filename))
    ]
    for metrics_path in candidate_paths:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                return json.load(f)
    return None

def compute_classification_matrix(cm, labels):
    """Compute precision, recall, and f1-score matrix from confusion matrix."""
    cm = np.array(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)

    precision = tp / np.maximum(predicted, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-10)

    rows = []
    for i, name in enumerate(labels):
        display_label = name.replace("_", " ").title()
        rows.append({
            "Category": display_label,
            "Precision": f"{precision[i]*100:.2f}%",
            "Recall": f"{recall[i]*100:.2f}%",
            "F1-Score": f"{f1[i]*100:.2f}%",
            "Support": int(support[i])
        })

    # Summary rows
    macro_row = {
        "Category": "**Macro Avg**",
        "Precision": f"{precision.mean()*100:.2f}%",
        "Recall": f"{recall.mean()*100:.2f}%",
        "F1-Score": f"{f1.mean()*100:.2f}%",
        "Support": int(support.sum())
    }
    weighted_f1 = (f1 * support).sum() / support.sum()
    weighted_prec = (precision * support).sum() / support.sum()
    weighted_rec = (recall * support).sum() / support.sum()
    weighted_row = {
        "Category": "**Weighted Avg**",
        "Precision": f"{weighted_prec*100:.2f}%",
        "Recall": f"{weighted_rec*100:.2f}%",
        "F1-Score": f"{weighted_f1*100:.2f}%",
        "Support": int(support.sum())
    }

    df = pd.DataFrame(rows + [macro_row, weighted_row])
    return df

def evaluation_tab(sentiment_pipeline):
    st.header("🔬 Deep Learning Model Evaluation & Metrics")
    st.caption("Quantitative performance, confusion matrices, and ablation analysis across benchmark research datasets.")

    dataset_choice = st.radio(
        "Choose Research Benchmark Dataset:",
        [
            "MindCare Emotion & Well-Being Dataset (6 Classes: Sadness, Joy, Love, Anger, Fear, Surprise)",
            "Reddit Mental Health Disorder Detection Dataset (Binary: Mental Distress vs Control)"
        ],
        horizontal=True
    )

    is_reddit = "Reddit" in dataset_choice
    metrics = load_metrics_data("reddit" if is_reddit else "emotion")
    
    if is_reddit:
        labels = ["control", "mental_distress"]
        model_keys = ["reddit_proposed", "reddit_baseline2", "reddit_baseline1"]
        dataset_desc = "Reddit Mental Health Corpus (Hasan et al., Advancing Mental Disorder Detection) — Binary Classification of Mental Distress vs Control Posts."
    else:
        labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        model_keys = ["proposed", "baseline2", "baseline1", "additional"]
        dataset_desc = "Saravia et al. Multi-Class Emotion & Mental Well-Being Benchmark Dataset (EMNLP)."

    st.info(f"📊 **Active Benchmark**: {dataset_desc}")

    if not metrics:
        st.warning(f"Metrics data for this dataset not found or still generating. Please verify that experiments have run in `model_experiments`.")
        return

    # 1. Ablation Overview Matrix
    st.subheader("1. Architecture Ablation Comparison Matrix")
    ablation_rows = []
    for k in model_keys:
        if k in metrics:
            m = metrics[k]
            ablation_rows.append({
                "Architecture": m.get("display_name", k),
                "Test Accuracy": f"{m.get('test_accuracy', 0)*100:.2f}%",
                "Macro F1": f"{m.get('test_macro_f1', 0)*100:.2f}%",
                "Weighted F1": f"{m.get('test_weighted_f1', 0)*100:.2f}%",
                "Macro Precision": f"{m.get('test_precision_macro', 0)*100:.2f}%",
                "Macro Recall": f"{m.get('test_recall_macro', 0)*100:.2f}%",
                "Inference Latency": f"{m.get('test_latency_ms', 0):.2f} ms"
            })
    
    if ablation_rows:
        st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True)

    st.markdown("---")

    # 2. Model Selector for Detailed Matrix
    st.subheader("2. Detailed Classification Matrix & Confusion Matrix")
    available_keys = [k for k in model_keys if k in metrics]
    if not available_keys:
        st.warning("No model evaluations available yet for this dataset.")
        return

    selected_key = st.selectbox(
        "Select Model to Inspect:",
        options=available_keys,
        format_func=lambda k: metrics[k].get("display_name", k)
    )

    model_info = metrics[selected_key]
    cm = model_info.get("test_confusion_matrix", [])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"#### Classification Report Matrix ({model_info.get('display_name')})")
        report_df = compute_classification_matrix(cm, labels)
        st.table(report_df)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Accuracy", f"{model_info.get('test_accuracy', 0)*100:.1f}%")
        col_m2.metric("Macro F1", f"{model_info.get('test_macro_f1', 0)*100:.2f}%")
        col_m3.metric("Latency", f"{model_info.get('test_latency_ms', 0):.1f} ms")

    with col2:
        st.markdown("#### Test Confusion Matrix (Counts & Heatmap)")
        cm_array = np.array(cm)
        
        # Normalized matrix for heatmap display
        cm_norm = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=[l.capitalize() for l in labels],
            y=[l.capitalize() for l in labels],
            text=cm_array,
            texttemplate="%{text}<br>(%{z:.1%})",
            colorscale='Blues',
            colorbar=dict(title='Recall')
        ))
        fig.update_layout(
            title=f"Confusion Matrix — {model_info.get('display_name')}",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            yaxis_autorange='reversed',
            height=450,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3. Live Evaluation Tester with Attention Explainability
    st.subheader("3. Live Evaluation & Token Attention Inspector")
    st.markdown("Test the deployed model on any sample sentence or clinical statement to inspect live attention weights:")

    sample_prompts = [
        "I feel completely overwhelmed and hopeless with everything going on.",
        "Had a wonderful time celebrating with close friends and feeling very calm.",
        "I am really terrified about failing my upcoming test tomorrow.",
        "I am so frustrated and angry with how unfair this situation has been.",
        "I was genuinely surprised to hear from my old friend today."
    ]

    selected_sample = st.selectbox("Or choose a sample journal entry:", ["Custom entry..."] + sample_prompts)
    default_text = "" if selected_sample == "Custom entry..." else selected_sample
    eval_text = st.text_area("Input text to evaluate:", value=default_text, height=90)

    if st.button("Run Live Evaluation", type="primary"):
        if eval_text.strip() and sentiment_pipeline:
            # Run prediction
            res = sentiment_pipeline(eval_text)[0]
            pred_emotion = res.get('label', 'neutral')
            confidence = res.get('score', 0.5)
            sentiment = res.get('sentiment', 'NEUTRAL')
            all_probs = res.get('all_probabilities', {})
            top_words = res.get('top_attended_words', [])

            st.success(f"**Predicted Emotion:** {pred_emotion.capitalize()} | **Sentiment:** {sentiment} | **Confidence:** {confidence*100:.1f}%")

            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.markdown("##### Class Probability Distribution")
                if all_probs:
                    probs_df = pd.DataFrame([
                        {"Emotion": k.capitalize(), "Probability": v}
                        for k, v in all_probs.items()
                    ]).sort_values("Probability", ascending=True)

                    bar_fig = px.bar(
                        probs_df,
                        x="Probability",
                        y="Emotion",
                        orientation="h",
                        color="Probability",
                        color_continuous_scale="Purples",
                        text=probs_df["Probability"].apply(lambda p: f"{p*100:.1f}%")
                    )
                    bar_fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                    st.plotly_chart(bar_fig, use_container_width=True)

            with col_res2:
                st.markdown("##### Attention Mechanism (Slide 15 Explainability)")
                if top_words:
                    st.write("Tokens receiving the highest additive attention weights:")
                    for idx, word in enumerate(top_words, 1):
                        st.markdown(f"**{idx}.** `{word}`")
                    st.caption("The proposed model prioritizes these specific tokens to determine emotional distress.")
                else:
                    st.info("Attention tokens available when running the Proposed DistilBERT + BiGRU + Attention model.")
        elif not eval_text.strip():
            st.warning("Please enter some text to evaluate.")
