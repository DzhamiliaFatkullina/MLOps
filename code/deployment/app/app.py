import streamlit as st
import os
import json
import requests
from pathlib import Path

def load_metadata():
    md = {}
    md_path = Path("models/metadata.json")
    if md_path.exists():
        md = json.load(open(md_path))
    else:
        # fallback to a simple call to API root
        try:
            api_root = os.environ.get("API_URL", "http://api:8000")
            r = requests.get(f"{api_root}/")
            if r.ok:
                md = {"feature_names": [], "target_names": []}
        except:
            md = {"feature_names": [], "target_names": []}
    return md

st.set_page_config(page_title="Simple MLOps App")
st.title("Simple MLOps demo")

metadata = load_metadata()
feature_count = len(metadata.get("feature_names", []))

st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", value=os.environ.get("API_URL", "http://api:8000"))
st.sidebar.write(f"Detected features: {feature_count}")
st.sidebar.write(f"Targets: {metadata.get('target_names', [])}")

st.write("Enter model features as comma-separated floats (same order as the training dataset).")
default = ",".join(["0"] * feature_count) if feature_count > 0 else "0"

features_text = st.text_area("Features", value=default, height=120)

if st.button("Predict"):
    try:
        feats = [float(x.strip()) for x in features_text.split(",") if x.strip() != ""]
        if feature_count and len(feats) != feature_count:
            st.error(f"Model expects {feature_count} features, got {len(feats)}")
        else:
            resp = requests.post(f"{api_url}/predict", json={"features": feats})
            resp.raise_for_status()
            st.success("Result:")
            st.json(resp.json())
    except Exception as e:
        st.error(str(e))
