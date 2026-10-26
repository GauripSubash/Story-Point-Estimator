"""Local research demo. This app loads the frozen model; it never retrains it."""
import streamlit as st
from src.common import ROOT, read_json
from src.predict import load_model, predict_ticket

st.set_page_config(page_title="Story Point Estimator", page_icon="\u25a6", layout="centered")
st.title("Story Point Estimator")
st.write("Turn a ticket description into a starting point for your team’s estimation discussion.")
st.caption("Local research reconstruction • TF-IDF text classification • TAWOS")

@st.cache_resource
def saved_model():
    return load_model()

model_path = ROOT / "models/selected.joblib"
if not model_path.exists():
    st.error("The saved model is missing. Restore models/selected.joblib from this frozen experiment, then restart this demo.")
    st.stop()

with st.form("ticket"):
    title = st.text_input("Ticket title", placeholder="Add retry handling for failed webhook deliveries")
    description = st.text_area("Description", placeholder="Describe the requested behavior, constraints, and acceptance criteria.", height=180)
    submit = st.form_submit_button("Estimate category", type="primary")

if submit:
    try:
        result = predict_ticket(title, description, model=saved_model())
    except (ValueError, FileNotFoundError) as error:
        st.warning(str(error))
    else:
        st.subheader(result["category"].capitalize())
        st.write(result["range"])
        st.info(result["note"])

with st.expander("How to interpret this estimate"):
    st.write("Low: (0, 2] points. Medium: (2, 5]. High: (5, 100] in the training data. Points express a team’s relative view of effort, complexity, and uncertainty; they are not hours.")
    st.write("The model uses ticket text from historical open-source projects. Your team may estimate differently. Later edits may remain in source descriptions despite filtering; this limits how well an offline score represents real estimation.")
    st.write("No confidence percentage is shown: model scores have not been calibrated. Use the category to support discussion, then make your own estimate.")
    metrics_path = ROOT / "reports/test_metrics.json"
    if metrics_path.exists():
        report = read_json(metrics_path)
        scores = report["models"][report["selected_family"]]
        baseline = report["models"]["baseline"]
        st.caption(f"Frozen chronological test: {scores['test_size']:,} tickets; category accuracy {scores['accuracy']:.1%}; macro F1 {scores['macro_f1']:.3f}. Majority baseline accuracy: {baseline['accuracy']:.1%}.")
