import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import re
from wordcloud import WordCloud
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from streamlit_option_menu import option_menu
import plotly.graph_objects as go

import nltk
import os

# Folder khusus untuk menyimpan resource NLTK
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download stopwords dan resource WordNet jika belum ada
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)


# ========== Utility Functions ==========

def preprocess_text(text):
    stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text.lower())
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    labels = sorted(list(set(list(y_true) + list(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    return fig

def generate_wordcloud(text, title=None):
    if not text.strip():
        st.write("No data available to generate WordCloud.")
        return
    wc = WordCloud(
        background_color='white',
        max_words=100,
        width=400,
        height=300,
        contour_width=1,
        contour_color='steelblue',
        collocations=False,
        stopwords=set(stopwords.words('indonesian') + stopwords.words('english'))
    ).generate(text)
    plt.figure(figsize=(6, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title)
    st.pyplot(plt)

# ========== Cache Loaders ==========

@st.cache_data(show_spinner=True)
def load_data():
    return pd.read_csv('data/data_dengan_sentimen.csv')

@st.cache_resource(show_spinner=True)
def load_models():
    models = {}
    try:
        models['nb'] = joblib.load('model/nb_pipeline.pkl')
        models['svm'] = joblib.load('model/svm_pipeline.pkl')
        models['xgb'] = joblib.load('model/xgb_pipeline.pkl')
        models['tfidf'] = joblib.load('model/tfidf_vectorizer.pkl')
        models['label_encoder'] = joblib.load('model/label_encoder.pkl')
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

# ========== PAGE CONFIG ==========

st.set_page_config(
    page_title="Analyzing Mental Health Sentiment on X App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== SIDEBAR & APP STYLING ==========

st.markdown("""
    <style>
    /* ========== BACKGROUND SOLID FORMAL ========== */
    .stApp {
        background-color: #f0f4f8 !important; /* biru keabu-abuan yang lembut */
        color: #1A1A1A !important;
    }

    /* ========== SIDEBAR TRANSPARAN ELEGAN ========== */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background-color: rgba(255, 255, 255, 0.4) !important;
        box-shadow: none !important;
        border: none !important;
    }
    .stSidebarContent {
        background: transparent !important;
        padding: 0 !important;
    }
    .css-1lcbmhc.e1fqkh3o3 {  /* class sidebar menu item */
        font-size: 13px !important;  /* ubah sesuai keinginan, misalnya 13px */
    }
    .css-1wvake5.e1fqkh3o3 {  /* class sidebar selected item */
        font-size: 13px !important;

       div[data-baseweb="select"] > div {
        font-size: 13px !important;
    } 

    /* ========== MENU NAVIGASI (OPTION MENU) ========== */
    ul.nav.nav-pills > li > a {
        background-color: rgba(220, 230, 250, 0.3) !important;
        color: #1A1A1A !important;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }

    ul.nav.nav-pills > li > a.active {
        background-color: rgba(100, 149, 237, 0.6) !important; /* cornflowerblue */
        color: white !important;
    }

    ul.nav.nav-pills > li > a:hover {
        background-color: rgba(173, 216, 230, 0.5) !important;
        color: #1A1A1A !important;
    }

    /* ========== TOMBOL ========== */
    div.stButton > button {
        background-color: rgba(110, 231, 183, 0.4) !important;
        color: #1A1A1A !important;
        border: 1px solid #1A1A1A !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: rgba(110, 231, 183, 0.7) !important;
        color: white !important;
        border-color: white !important;
    }

    /* ========== INPUT FIELD ========== */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.4) !important;
        color: #1A1A1A !important;
        border: 1px solid #1A1A1A !important;
    }

    .stFileUploader > div {
        background-color: rgba(255, 255, 255, 0.4) !important;
        border: 1px solid #1A1A1A !important;
        color: #1A1A1A !important;
    }

    /* ========== KOMPONEN UTAMA ========== */
    .st-cb, .stDataFrame, .stTable, .stAlert, .stExpander,
    .stMarkdown, .stCard, .stCaption, .element-container {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
    }

    /* ========== WARNA TEKS UTAMA ========== */
    .stApp, .stMarkdown, .stText, .stHeader, .stSubheader,
    .stCaption, .stDataFrame, .stTable, .stExpander, .stCard {
        color: #1A1A1A !important;
    }

    /* ========== FOOTER ========== */
    .footer {
        background-color: #333 !important;
        color: white !important;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }

    .footer a {
        color: #00b6ff;
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# ========== SIDEBAR HEADER ==========

with st.sidebar:
    st.markdown("""
        <div style="
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 10px;
        ">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ce/X_logo_2023.svg" width="50" style="margin: 0;"/>
            <h2 style="color: #000000; font-weight: bold; margin: 0; font-size: 20px;">
                Analyzing Mental Health<br>Sentiment on X App
            </h2>
        </div>
    """, unsafe_allow_html=True)
with st.sidebar:
    selected_tab = option_menu(
        menu_title="Menu",
        options=[
            "Sentiment Analyze",
            "Comparison Algorithm",
            "Sentiment Prediction"
        ],
        icons=["bar-chart", "activity", "search"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
        
    )

# ========== Load Data & Models ==========

data = load_data()
models = load_models()

nb_model = models.get('nb')
svm_model = models.get('svm')
xgb_model = models.get('xgb')
tfidf_vectorizer = models.get('tfidf')
le = models.get('label_encoder')

# ========== Split Data ==========

train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data['Sentiment']
)
X_test_raw = test_data['tweet']
y_test = test_data['Sentiment']

# ========== PAGE LOGIC ==========

if selected_tab == "Sentiment Analyze":
    st.subheader("Sentiment Analyze")

    sentiment_counts = data['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    # Mapping label ke format baku
    label_mapping = {
        "positif": "Positif",
        "negatif": "Negatif",
        "netral": "Netral"
    }

    # Bersihkan label → lowercase, strip spasi
    sentiment_counts["Sentiment"] = sentiment_counts["Sentiment"].astype(str).str.strip().str.lower()

    # Map label ke format standar
    sentiment_counts["Sentiment"] = sentiment_counts["Sentiment"].map(label_mapping)

    # Hapus baris kosong (jika ada label tidak dikenali)
    sentiment_counts = sentiment_counts.dropna(subset=["Sentiment"])

    # Urutkan label sesuai mapping warna
    sentiment_order = ["Positif", "Negatif", "Netral"]
    sentiment_counts = (
        sentiment_counts
        .set_index("Sentiment")
        .reindex(sentiment_order)
        .fillna(0)
        .reset_index()
    )
    color_map = {
        "Positif": "#2ecc71",   
        "Negatif": "#e74c3c",   
        "Netral": "#3498db"    
    }

    col1, col2 = st.columns(2)

    with col1:
        fig_bar = px.bar(
            sentiment_counts,
            x='Sentiment',
            y='Count',
            color='Sentiment',
            title="Bar Chart of Sentiment Distribution",
            color_discrete_map=color_map
        )
        fig_bar.update_layout(
            width=400,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#000000", size=16),
            title_font=dict(size=18, color="#000000"),
            legend=dict(font=dict(color="#000000")),
            xaxis=dict(
                title_font=dict(color="#000000", size=14),
                tickfont=dict(color="#000000", size=12)
            ),
            yaxis=dict(
                title_font=dict(color="#000000", size=14),
                tickfont=dict(color="#000000", size=12)
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            sentiment_counts,
            values='Count',
            names='Sentiment',
            hole=0.4,
            title="Sentiment Proportion",
            color='Sentiment',                    
            color_discrete_map=color_map
        )
        fig_pie.update_layout(
            width=400,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#000000", size=16),
            title_font=dict(size=18, color="#000000"),
            legend=dict(font=dict(color="#000000"))
        )
        fig_pie.update_traces(
            textinfo='percent+label',
            textfont_color='#000000'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    freq = pd.Series(' '.join(data['tweet']).split()).value_counts().head(20)
    fig_freq = px.bar(
        freq,
        x=freq.index,
        y=freq.values,
        labels={'x': 'Words', 'y': 'Frequency'},
        title="Top 20 Words by Frequency"
    )

    fig_freq.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#000000", size=16),
        title_font=dict(size=20, color="#000000"),
        legend=dict(font=dict(color="#000000")),
        xaxis=dict(
            title_font=dict(color='#000000', size=16),
            tickfont=dict(color='#000000', size=14)
        ),
        yaxis=dict(
            title_font=dict(color='#000000', size=16),
            tickfont=dict(color='#000000', size=14)
        )
    )

    fig_freq.update_traces(textfont_color='#000000')

    st.plotly_chart(fig_freq, use_container_width=True)

    st.markdown("### WordCloud per Sentiment")
    sentiments = ['Positif', 'Negatif', 'Netral']
    cols_wc = st.columns(len(sentiments))
    for i, sent in enumerate(sentiments):
        subset_text = ' '.join(data[data['Sentiment'] == sent]['tweet'].apply(preprocess_text))
        with cols_wc[i]:
            st.markdown(f"#### {sent}")
            generate_wordcloud(subset_text)

elif selected_tab == "Comparison Algorithm":
    st.header("Comparison Algorithm")

    models_dict = {
        "Naive Bayes": nb_model,
        "Support Vector Machine": svm_model,
        "XGBoost": xgb_model
    }

    known_labels = set(le.classes_)
    mask = y_test.isin(known_labels)
    X_test_raw_filtered = X_test_raw[mask]
    y_test_filtered = y_test[mask]

    X_test_prep = X_test_raw_filtered.apply(preprocess_text)

    metrics_summary = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-score": []
    }

    cols_models = st.columns(3)

    model_results = {}

    for i, (model_name, model) in enumerate(models_dict.items()):
        with cols_models[i]:
            st.markdown(f"### {model_name}")

            if model is None:
                st.warning("Model not loaded.")
                continue

            y_pred = model.predict(X_test_prep)
            y_pred_encoded = le.transform(y_pred) if isinstance(y_pred[0], str) else y_pred
            y_test_encoded = le.transform(y_test_filtered)
            y_true_str = le.inverse_transform(y_test_encoded)
            y_pred_str = le.inverse_transform(y_pred_encoded)

            acc = accuracy_score(y_true_str, y_pred_str)
            report = classification_report(
                y_true_str, y_pred_str,
                labels=["Positif", "Negatif", "Netral"],
                output_dict=True,
                zero_division=0
            )
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1score = report['weighted avg']['f1-score']

            metrics_summary["Model"].append(model_name)
            metrics_summary["Accuracy"].append(acc)
            metrics_summary["Precision"].append(precision)
            metrics_summary["Recall"].append(recall)
            metrics_summary["F1-score"].append(f1score)

            fig_cm = plot_confusion_matrix(y_true_str, y_pred_str, f"{model_name}")
            st.pyplot(fig_cm)

            with st.expander("Report"):
                # Tampilkan classification report table
                report_df = pd.DataFrame(report).transpose()
                report_df = report_df.round(2)  # Membulatkan semua nilai numerik ke 2 desimal
                st.dataframe(report_df)


                # Overall accuracy & jumlah data uji
                overall_acc = round(report["accuracy"], 2)  # dibulatkan ke 2 desimal
                n_test_data = len(y_true_str)

                st.markdown(f"""
                    <div style="
                        background-color: #ffe6e6;
                        border-left: 5px solid #ff4d4d;
                        padding: 10px;
                        margin-top: 10px;
                        color: black;
                        ">
                        <b>Overall Accuracy:</b> {overall_acc:.2f}<br>
                        <b>Number of Test Data:</b> {n_test_data}
                        </div>
                    """, unsafe_allow_html=True)


                # Klasifikasi hasil algoritma
                pred_counts = pd.Series(y_pred_str).value_counts().to_dict()
                st.markdown("<h4>Classification Result Summary:</h4>", unsafe_allow_html=True)

                for label, count in pred_counts.items():
                    st.markdown(f"- <b>{label}:</b> {count} data", unsafe_allow_html=True)

                # Kesimpulan sederhana
                conclusion = ""
                if overall_acc >= 0.85:
                    conclusion = "Model ini memiliki performa yang sangat baik untuk analisis sentimen."
                elif overall_acc >= 0.7:
                    conclusion = "Model ini memiliki performa yang cukup baik, namun masih bisa ditingkatkan."
                else:
                    conclusion = "Model ini memiliki performa yang kurang optimal dan perlu perbaikan."

                st.markdown(f"""
                    <div style="
                        background-color: #ccf5ff;
                        border-left: 5px solid #007acc;
                        padding: 10px;
                        margin-top: 10px;
                        color: black;
                    ">
                        <b>Kesimpulan:</b><br>{conclusion}
                    </div>
                """, unsafe_allow_html=True)

            # Simpan hasil akurasi untuk rekomendasi
            model_results[model_name] = round(overall_acc, 2)

    st.markdown("---")
    st.subheader("Grouped Stacked Bar Chart")

    if metrics_summary["Model"]:
        metrics_df = pd.DataFrame(metrics_summary).set_index("Model")

        fig = go.Figure()

        for metric in metrics_df.columns:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=metrics_df.index,
                    y=metrics_df[metric],
                    text=metrics_df[metric].round(2),
                    textposition='auto',
                    opacity=0.8
                )
            )

        fig.update_layout(
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=16),
            xaxis=dict(
                title='Model',
                color='white',
                tickangle=-45
            ),
            yaxis=dict(
                title='Metric Values',
                color='white'
            ),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14)
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- NEW: Rekomendasi Model ---
        best_model_name = max(model_results, key=model_results.get)
        best_model_acc = model_results[best_model_name]

        st.markdown(f"""
            <div style="
                background-color: #d4edda;
                border-left: 5px solid #28a745;
                padding: 10px;
                margin-top: 20px;
                color: black;
            ">
                <b>🔍 Rekomendasi Model:</b><br>
                Model yang direkomendasikan adalah <b>{best_model_name}</b> dengan akurasi {best_model_acc:.2f}.
            </div>
        """, unsafe_allow_html=True)

    else:
        st.info("No evaluation metrics available.")


elif selected_tab == "Sentiment Prediction":
    
    st.subheader("Sentiment Prediction")

    if not xgb_model:
        st.error("XGBoost model not loaded.")
    else:
        new_text = st.text_input("Enter text for sentiment prediction:")

        if st.button("Predict Sentiment"):
            if not new_text.strip():
                st.warning("Please enter some text to predict.")
            else:
                preprocessed = preprocess_text(new_text)
                try:
                    prediction = xgb_model.predict([preprocessed])
                    if le:
                        predicted_sentiment = le.inverse_transform(prediction)
                    else:
                        predicted_sentiment = prediction
                    st.success(f"**Predicted Sentiment:** {predicted_sentiment[0].capitalize()}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        st.markdown("### 📂 Predict Sentiment from File (.txt, one sentence per line)")
        uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])

        if uploaded_file is not None:
            lines = uploaded_file.read().decode('utf-8').split('\n')
            preprocessed_lines = [preprocess_text(line) for line in lines if line.strip()]
            try:
                predictions = xgb_model.predict(preprocessed_lines)
                if le:
                    decoded_preds = le.inverse_transform(predictions)
                else:
                    decoded_preds = predictions
                results_df = pd.DataFrame({
                    "Text": lines,
                    "Predicted Sentiment": [p.capitalize() for p in decoded_preds]
                })
                st.dataframe(results_df)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ========== FOOTER ==========

current_year = datetime.now().year
st.markdown(f"""
    <style>
    .footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #333;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        z-index: 1000;
    }}
    .footer a {{
        color: #00b6ff;
        text-decoration: none;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}
    </style>
    <div class="footer">
        Copyright © {current_year} | Apps Created by <b><a href="https://www.linkedin.com/in/sriagustin/" target="_blank">Sri Agustin</a></b>
    </div>
""", unsafe_allow_html=True)
