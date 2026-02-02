import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Iris Classifier", layout="wide")

if 'history' not in st.session_state:
    st.session_state['history'] = []

if not os.path.exists('iris_model.pkl') or not os.path.exists('model_metrics.pkl'):
    st.error("Error: Files missing! Please run 'train.py' first.")
else:
    model = joblib.load('iris_model.pkl')
    metrics = joblib.load('model_metrics.pkl')
    
    target_names = ['Setosa', 'Versicolor', 'Virginica']
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    flower_images = {
        'Setosa': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg',
        'Versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
        'Virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
    }

    st.title("🌸 Iris Species Classifier")
    st.markdown("---")

    st.sidebar.header("Adjust Flower Dimensions")
    s_l = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
    s_w = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
    p_l = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
    p_w = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

    input_data = [[s_l, s_w, p_l, p_w]]
    prediction_idx = model.predict(input_data)[0]
    result = target_names[prediction_idx]
    proba = model.predict_proba(input_data)

    current_entry = [s_l, s_w, p_l, p_w, result]
    if not st.session_state['history'] or st.session_state['history'][-1] != current_entry:
        st.session_state['history'].append(current_entry)

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Visual Analytics", "📜 History", "⚙️ Model Evaluation"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Classification Result")
            st.success(f"Predicted Species: **{result}**")
            st.metric("Model Confidence", f"{max(proba[0])*100:.2f}%")
        with col2:
            st.image(flower_images[result], caption=f"Iris {result}", use_container_width=True)

    with tab2:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Confidence Probability")
            fig_conf, ax_conf = plt.subplots()
            ax_conf.bar(target_names, proba[0], color=['#FF9999', '#66B2FF', '#99FF99'])
            st.pyplot(fig_conf)
        with col_g2:
            st.subheader("Feature Importance")
            importance = model.feature_importances_
            fig_imp, ax_imp = plt.subplots()
            ax_imp.barh(feature_names, importance, color='#4CAF50')
            st.pyplot(fig_imp)

    with tab3:
        st.subheader("Prediction History Log")
        if st.session_state['history']:
            history_df = pd.DataFrame(st.session_state['history'], columns=feature_names + ['Predicted Species'])
            st.dataframe(history_df.iloc[::-1], use_container_width=True)
           
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                csv = history_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download History (CSV)", csv, "history.csv", "text/csv")
            
            with btn_col2:
                if st.button("🗑️ Clear All History"):
                    st.session_state['history'] = []
                    st.rerun() 
        else:
            st.info("No history yet. Start moving sliders to see data here!")

    with tab4:
        st.subheader("Technical Performance & Metrics")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.write("### Model Accuracy")
            st.metric(label="Overall Test Accuracy", value=f"{metrics['accuracy']*100:.2f}%")
            st.write("### Model Details")
            st.write("- **Algorithm:** Random Forest Classifier")
            st.write("- **Dataset:** Iris Flower Dataset (Fisher, 1936)")
            st.write("- **Training Samples:** 120 (80%)")
            st.write("- **Testing Samples:** 30 (20%)")
        with m_col2:
            st.write("### Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(metrics['matrix'], annot=True, cmap='Blues', fmt='g',
                        xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
            st.pyplot(fig_cm)