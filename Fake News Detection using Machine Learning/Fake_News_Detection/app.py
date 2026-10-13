import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st

from preprocessing import preprocess_text, clean_text
from prediction import predict_news, load_model_and_vectorizer

st.set_page_config(
    page_title="AI Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Background and Fonts */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Gradient Title */
    .header-title {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Custom Card Style */
    .css-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
    }
    
    /* Metric Cards */
    .metric-box {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Result Badges */
    .badge-real {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }
    
    .badge-fake {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }
    
    /* Custom Sidebar Styling */
    .css-1d38152 {
        background-color: #161e2e;
    }
    
    /* Indicators pill */
    .indicator-pill-fake {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 3px;
    }
    
    .indicator-pill-real {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
COMPARISON_PATH = os.path.join(BASE_DIR, "models_comparison.pkl")
FAKE_CSV = os.path.join(BASE_DIR, "dataset", "Fake.csv")
TRUE_CSV = os.path.join(BASE_DIR, "dataset", "True.csv")

if 'history' not in st.session_state:
    st.session_state.history = []
if 'input_headline' not in st.session_state:
    st.session_state.input_headline = ""
if 'input_article' not in st.session_state:
    st.session_state.input_article = ""


@st.cache_data
def load_dataset_cached():
    if os.path.exists(FAKE_CSV) and os.path.exists(TRUE_CSV):
        df_fake = pd.read_csv(FAKE_CSV)
        df_true = pd.read_csv(TRUE_CSV)
        df_fake['label'] = 0
        df_true['label'] = 1
        df = pd.concat([df_fake, df_true], ignore_index=True)
        return df
    return None


@st.cache_data
def load_metrics_cached():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


st.sidebar.image("https://img.icons8.com/color/96/000000/news.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page:",
    [
        "📰 Predict News",
        "📊 Dashboard Analytics",
        "🧪 Model Evaluation",
        "🎓 About & Resume Info"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📜 Recent Predictions")
if st.session_state.history:
    for idx, item in enumerate(reversed(st.session_state.history[-5:])):
        label_color = "#10b981" if item['is_real'] else "#ef4444"
        st.sidebar.markdown(
            f"**{idx+1}. {item['headline'][:25]}...**  \n"
            f"<span style='color:{label_color}; font-weight:bold;'>{item['label']}</span> ({item['confidence']}%)",
            unsafe_allow_html=True
        )
else:
    st.sidebar.info("No predictions yet in this session.")

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()


if page == "📰 Predict News":
    st.markdown('<div class="header-title">AI-Based Fake News Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Analyze news headlines and article body text in real time using Natural Language Processing & Machine Learning</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        st.markdown("### 📝 Enter News Details")
        
        headline = st.text_input(
            "News Headline",
            placeholder="e.g., Senate passes bipartisan infrastructure bill...",
            key="input_headline_field"
        )
        
        article_text = st.text_area(
            "News Article Body Content",
            placeholder="Paste full news article text here...",
            height=200,
            key="input_article_field"
        )

        col_btn1, col_btn2, _ = st.columns([1, 1, 2])
        with col_btn1:
            predict_clicked = st.button("🔍 Predict News", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🧹 Clear Inputs", use_container_width=True):
                st.session_state.input_headline_field = ""
                st.session_state.input_article_field = ""
                st.rerun()

        st.markdown("---")
        st.markdown("#### 💡 Quick Test Examples")
        c1, c2 = st.columns(2)
        if c1.button("🧪 Try Fake News Sample"):
            st.session_state.input_headline_field = "SHOCKING: Hidden Alien Artifact Discovered Beneath Antarctic Ice Sheet!"
            st.session_state.input_article_field = "BOMBSHELL report! Whistleblowers reveal top secret underground bunker hiding extra-terrestrial technology away from public oversight."
            st.rerun()
            
        if c2.button("🧪 Try Real News Sample"):
            st.session_state.input_headline_field = "Federal Reserve Keeps Interest Rates Steady Amid Economic Indicators"
            st.session_state.input_article_field = "WASHINGTON (Reuters) - Official representatives confirmed today that the legislation was approved following extensive committee negotiations."
            st.rerun()

    with col_right:
        st.markdown("### 🎯 Classification Result")
        
        if predict_clicked:
            result = predict_news(headline, article_text)
            
            if result['status'] == "error":
                st.error(result['message'])
            else:
                # Add to history
                st.session_state.history.append({
                    'headline': headline if headline else article_text[:30],
                    'label': result['label'],
                    'confidence': result['confidence'],
                    'is_real': result['is_real']
                })
                
                if result['is_real']:
                    st.markdown(f'<div class="badge-real">✅ {result["label"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-fake">🚨 {result["label"]}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f"**Confidence Level:** `{result['confidence']}%`")
                st.progress(result['confidence'] / 100.0)
                
                c_real, c_fake = st.columns(2)
                c_real.metric("Real Probability", f"{result['real_probability']}%")
                c_fake.metric("Fake Probability", f"{result['fake_probability']}%")
                
                with st.expander("🔍 View Preprocessed NLP Tokens"):
                    st.code(result['preprocessed_snippet'], language='text')

                st.markdown("#### 🧠 Key Influencing Keywords")
                indicators = result['indicators']
                
                if result['is_real']:
                    if indicators['real_indicators']:
                        st.markdown("**Real News Indicators Detected:**")
                        html_pills = "".join([f'<span class="indicator-pill-real">{w}</span>' for w in indicators['real_indicators']])
                        st.markdown(html_pills, unsafe_allow_html=True)
                    else:
                        st.write("Neutral phrase patterns detected.")
                else:
                    if indicators['fake_indicators']:
                        st.markdown("**Sensational/Fake Indicators Detected:**")
                        html_pills = "".join([f'<span class="indicator-pill-fake">{w}</span>' for w in indicators['fake_indicators']])
                        st.markdown(html_pills, unsafe_allow_html=True)
                    else:
                        st.write("Pattern matching fake narrative templates.")
        else:
            st.info("👈 Enter news details on the left and click **Predict News** to view AI classification results.")


elif page == "📊 Dashboard Analytics":
    st.markdown('<div class="header-title">Dataset Overview & NLP Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Statistical insights, distribution metrics, and word cloud visualizer for real vs fake news corpus</div>', unsafe_allow_html=True)

    metrics_data = load_metrics_cached()
    df = load_dataset_cached()

    if metrics_data and df is not None:
        summary = metrics_data['dataset_summary']
        
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-box"><div class="metric-value">{summary["total_articles"]}</div><div class="metric-label">Total Articles</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#ef4444">{summary["fake_count"]}</div><div class="metric-label">Fake Articles</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#10b981">{summary["real_count"]}</div><div class="metric-label">Real Articles</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_pie, col_sub = st.columns([1, 1.2])
        
        with col_pie:
            st.markdown("### 🥧 Label Distribution")
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5), facecolor='#0e1117')
            labels = ['Fake News', 'Real News']
            sizes = [summary['fake_count'], summary['real_count']]
            colors = ['#ef4444', '#10b981']
            
            wedges, texts, autotexts = ax_pie.pie(
                sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=140, textprops=dict(color="w", weight="bold")
            )
            ax_pie.axis('equal')
            st.pyplot(fig_pie)

        with col_sub:
            st.markdown("### 🏷️ News Category / Subject Distribution")
            if 'subject' in df.columns:
                fig_sub, ax_sub = plt.subplots(figsize=(6, 4.5), facecolor='#0e1117')
                ax_sub.set_facecolor('#1e293b')
                subject_counts = df['subject'].value_counts()
                sns.barplot(x=subject_counts.values, y=subject_counts.index, palette="mako", ax=ax_sub)
                ax_sub.set_title("Articles count by subject category", color="white")
                ax_sub.tick_params(colors="white")
                for spine in ax_sub.spines.values():
                    spine.set_color('#334155')
                st.pyplot(fig_sub)
            else:
                st.info("Subject categories not present in merged view.")

        st.markdown("---")
        st.markdown("### ☁️ Word Clouds (NLP Corpus Exploration)")
        
        col_wc1, col_wc2 = st.columns(2)
        
        df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        fake_sample_text = " ".join(df[df['label']==0]['combined'].sample(min(300, len(df[df['label']==0])), random_state=42).apply(clean_text))
        real_sample_text = " ".join(df[df['label']==1]['combined'].sample(min(300, len(df[df['label']==1])), random_state=42).apply(clean_text))

        with col_wc1:
            st.markdown("#### 🚨 Fake News Word Cloud")
            wc_fake = WordCloud(width=600, height=350, background_color='#1e293b', colormap='Reds').generate(fake_sample_text)
            fig_wc1, ax_wc1 = plt.subplots(figsize=(6, 3.5), facecolor='#0e1117')
            ax_wc1.imshow(wc_fake, interpolation='bilinear')
            ax_wc1.axis('off')
            st.pyplot(fig_wc1)

        with col_wc2:
            st.markdown("#### ✅ Real News Word Cloud")
            wc_real = WordCloud(width=600, height=350, background_color='#1e293b', colormap='Greens').generate(real_sample_text)
            fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 3.5), facecolor='#0e1117')
            ax_wc2.imshow(wc_real, interpolation='bilinear')
            ax_wc2.axis('off')
            st.pyplot(fig_wc2)

    else:
        st.error("Metrics or dataset file missing. Please ensure dataset and train_model.py have been executed.")

elif page == "🧪 Model Evaluation":
    st.markdown('<div class="header-title">Machine Learning Model Performance & Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Detailed evaluation comparing Logistic Regression, Naive Bayes, and Random Forest algorithms</div>', unsafe_allow_html=True)

    metrics_data = load_metrics_cached()
    
    if metrics_data:
        models_perf = metrics_data['model_metrics']
        
        st.markdown("### 📊 Model Performance Comparison Table")
        
        table_rows = []
        for model_name, m in models_perf.items():
            table_rows.append({
                "Model Algorithm": model_name,
                "Accuracy (%)": f"{m['accuracy']*100:.2f}%",
                "Precision (%)": f"{m['precision']*100:.2f}%",
                "Recall (%)": f"{m['recall']*100:.2f}%",
                "F1 Score (%)": f"{m['f1_score']*100:.2f}%",
                "ROC AUC Score": f"{m['roc_auc']:.4f}"
            })
            
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True)

        st.markdown("---")
        
        selected_model = st.selectbox("Select Model to Inspect Visualizations:", list(models_perf.keys()))
        m_details = models_perf[selected_model]

        c_cm, c_roc = st.columns(2)
        
        with c_cm:
            st.markdown(f"#### 🌀 Confusion Matrix ({selected_model})")
            cm = np.array(m_details['confusion_matrix'])
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake (0)', 'Real (1)'],
                yticklabels=['Fake (0)', 'Real (1)'],
                ax=ax_cm, cbar=False
            )
            ax_cm.set_ylabel('True Label', color='white')
            ax_cm.set_xlabel('Predicted Label', color='white')
            ax_cm.tick_params(colors='white')
            st.pyplot(fig_cm)

        with c_roc:
            st.markdown(f"#### 📈 ROC Curve ({selected_model})")
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
            ax_roc.set_facecolor('#1e293b')
            
            if m_details['fpr'] and m_details['tpr']:
                ax_roc.plot(m_details['fpr'], m_details['tpr'], color='#38bdf8', lw=2, label=f"AUC = {m_details['roc_auc']:.4f}")
                ax_roc.plot([0, 1], [0, 1], color='#64748b', linestyle='--')
            
            ax_roc.set_xlabel('False Positive Rate', color='white')
            ax_roc.set_ylabel('True Positive Rate', color='white')
            ax_roc.tick_params(colors='white')
            ax_roc.legend(loc="lower right")
            for spine in ax_roc.spines.values():
                spine.set_color('#334155')
            st.pyplot(fig_roc)

        st.markdown("#### 📋 Detailed Classification Report")
        cls_df = pd.DataFrame(m_details['classification_report']).transpose()
        st.dataframe(cls_df.style.highlight_max(axis=0), use_container_width=True)

    else:
        st.error("Metrics JSON file missing. Please run train_model.py first.")

elif page == "🎓 About & Resume Info":
    st.markdown('<div class="header-title">About the Project & Portfolio Context</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Production-ready Machine Learning project designed for B.Tech AI & Data Science resumes and technical interviews</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Overview
    This **Fake News Detection System** leverages Natural Language Processing (NLP) techniques and Machine Learning classifiers to detect clickbait, exaggerated, or fabricated news articles.

    ---
    ### 🧠 Technical Pipeline & NLP Architecture
    1. **Data Ingestion**: Kaggle Fake & Real News Dataset integration (`Fake.csv` & `True.csv`).
    2. **NLP Text Preprocessing (`preprocessing.py`)**:
       - Lowercasing, HTML tag stripping, URL/Punctuation removal.
       - Tokenization & Stopwords removal (NLTK English corpus).
       - Stemming using PorterStemmer to reduce words to root form.
    3. **Feature Engineering**:
       - TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization with unigram & bigram features (`max_features=5000`).
    4. **Model Training & Comparison (`train_model.py`)**:
       - Primary Model: **Logistic Regression**
       - Comparative Models: **Multinomial Naive Bayes** & **Random Forest**
    5. **Explainable AI Insights (`prediction.py`)**:
       - Extracts key positive (Real) and negative (Fake) indicator words based on TF-IDF model feature weights.
    6. **Web Dashboard (`app.py`)**:
       - Streamlit responsive dashboard with color-coded classification, confidence gauge, dataset analytics, and word cloud visualization.

    ---
    ### 💼 Interview & Resume Highlight Points (AI/ML Fresher)
    * **Bullet Point 1**: Developed an end-to-end NLP Fake News Detection Web Application using Python, Scikit-learn, TF-IDF Vectorizer, and Streamlit.
    * **Bullet Point 2**: Applied custom text preprocessing (NLTK Stopwords removal & Porter Stemming), reducing feature space dimension while retaining semantic context.
    * **Bullet Point 3**: Evaluated Logistic Regression, Naive Bayes, and Random Forest models across Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
    * **Bullet Point 4**: Implemented explainable word feature attribution to interpret model predictions for real-time user inputs.
    """)
