import streamlit as st
import pandas as pd
import pickle
import plotly.express as px


model = pickle.load(open('toxic_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



if "history" not in st.session_state:
    st.session_state.history = []

def predict_toxicity(comment):
    comment_vectorized = vectorizer.transform([comment])
    probs = model.predict_proba(comment_vectorized)[0]
    
    result = dict(zip(labels, probs))
    final_labels = [labels[i] for i in range(len(labels)) if probs[i] > 0.3]
    
    max_prob = max(probs)
    if max_prob > 0.5:
        category = "Highly Toxic"
    elif max_prob > 0.3:
        category = "Neutral"
    else:
        category = "Safe"
    
    return result, final_labels, category
st.set_page_config(page_title="YouTube Comment Toxicity Detector", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
            .main {
                background-color: #f9f9f9;
            }
            .block-container {
                padding: 2rem;
            }
            st.TextArea {
            border: 1px solid #d9d9d9;
            border-radius: 12px;
            }
            [data-testid="stMetric"] {
            background:white;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.8);
            }
            </style>
""", unsafe_allow_html=True)
            


st.markdown("# YouTube Comment Toxicity Detector 🛡️")
st.caption("Enter a YouTube comment to analyze its toxicity level. The app will provide a toxicity score, confidence level, and predicted labels. Toxic comments are hidden to protect creators.")
user_input = st.text_area("Enter a YouTube comment:")

if user_input:
    result, final_labels, category = predict_toxicity(user_input)

    max_prob = max(result.values())
    confidence = round(max_prob * 100, 2)


    st.session_state.history.append({
    "comment": user_input,
    "score": round(max_prob, 2),
    "labels": final_labels,
    "category": category
     } )

# Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Toxicity score", round(max_prob, 2))
    col2.metric("Confidence", f"{confidence}%")
    col3.metric("Category", category)

# Graph
    st.subheader("Toxicity scores:")
    chart_df = pd.DataFrame({
       "category": list(result.keys()),
       "probability": list(result.values())
})

    fig = px.bar(
       chart_df,
       x="category",
       y="probability",
       color="category",
       title="Toxicity Scores"
)
    st.plotly_chart(fig, use_container_width=True)

# Prediction status
    st.subheader("Predicted labels:")
    if max_prob > 0.5:
        st.error(f"HIGHLY TOXIC COMMENT! Predicted labels: {', '.join(final_labels)}")
    elif max_prob > 0.3:
        st.warning(f"MODERATELY TOXIC COMMENT! Predicted labels: {', '.join(final_labels)}")
    else:
        st.success("SAFE COMMENT!")
   


st.markdown("## Creator Insights")

history_df = pd.DataFrame(st.session_state.history)

if not history_df.empty:
    safe = history_df[history_df['category'] == 'Safe']
    neutral = history_df[history_df['category'] == 'Neutral']
    toxic = history_df[history_df['category'] == 'Highly Toxic']

    c1, c2, c3 = st.columns(3)
    c1.metric("Safe comments", len(safe))
    c2.metric("Neutral comments", len(neutral))
    c3.metric("Highly toxic comments", len(toxic))

# Safe comments only
    st.markdown("## Safe comments (safe zone)")
    if not safe.empty:
       for _, row in safe.iterrows():
        st.success(f"""
Comment: {row['comment']}
Score: {round(row['score'], 2)}
""")
    else:
       st.write("No safe comments yet.")

    st.markdown("## Toxic comments (hidden)")
    if not toxic.empty:
       st.warning("Toxic comments are hidden to protect the creator.")
       for _, row in toxic.iterrows():
        st.error(f"""
🚫 Toxic comment hidden 📊 Score: {round(row['score'], 2)} 🏷 Labels: {', '.join(row['labels']) if row['labels'] else 'None'}
""")
    else:
      st.success("No toxic comments yet.")



st.markdown("## Live Comment Feed")

for item in reversed(st.session_state.history):
    score = item["score"]

    # Hide toxic text from feed too
    if score > 0.5:
        st.markdown(f"""
<div style="padding:10px; border:1px solid red; border-radius:5px; margin-bottom:10px; background-color:#ffe6e6;">
<b>🚫 Toxic comment hidden</b><br>
<b>Score:</b> {score}<br>
<b>Labels:</b> {', '.join(item['labels']) if item['labels'] else 'None'}
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style="padding:10px; border:1px solid green; border-radius:5px; margin-bottom:10px; background-color:#eaffea;">
<b>Comment:</b> {item['comment']}<br>
<b>Score:</b> {score}<br>
<b>Labels:</b> {', '.join(item['labels']) if item['labels'] else 'Safe'}
</div>
""", unsafe_allow_html=True)