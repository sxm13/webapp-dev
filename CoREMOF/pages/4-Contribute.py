import streamlit as st
import os, requests

from datetime import datetime



st.set_page_config(page_title="Contribute of CoRE MOF 2024 Database", page_icon="./figures/logo.png")

logo_url = "./figures/logo.png"
st.sidebar.image(logo_url)
st.logo(logo_url, link="https://zenodo.org/communities/core-mofs/records?q=&l=list&p=1&s=10&sort=newest")


st.title('Contribute a CIF')

with st.sidebar:
    st.markdown("© 2024 CoRE MOF Project. CC-BY-4.0 License.")

st.divider()

def create_directory(name):
    date_str = datetime.now().strftime("%Y%m%d")
    base_dir = "data/user/"
    i = 0
    while True:
        dir_path = os.path.join(base_dir, f"{date_str}{name}{i}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            return dir_path
        i += 1


slack_webhook_url = ""

name_cif = st.text_area("Name", "David Jones and Mary Johnson", height=68, key="name_cif")
email_cif = st.text_area("Email", "abc123@gmail.com", height=68, key="email_cif")
affi_cif = st.text_area("Institution", "School of Chemical Engineering, Pusan National University", height=68, key="affi_cif")
source = st.radio("Source", ["Experiment", "Simulation"], index=0, key="source_cif")
reference = st.text_area("Reference (DOI, title...)", height=68, key="reference_cif")
uploaded_file = st.file_uploader("Upload CIF file", type=['cif'], key="upload_cif")
agree = st.checkbox("I agree to share this CIF file.", key="agree_cif")
if st.button("📤 Upload CIF", key="submit_cif"):
    if not (name_cif and email_cif and affi_cif and source and reference):
        st.error("Please fill in your Name, Email, Institution, Source, and Reference before uploading.")
    elif not uploaded_file:
        st.error("Please upload a CIF file.")
    elif not agree:
        st.error("You must agree to share the CIF file.")
    else:
        dir_path = create_directory(name_cif)
        file_path = os.path.join(dir_path, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        message_details = f"New CIF file uploaded:\nName: {name_cif}\nEmail: {email_cif}\nInstitution: {affi_cif}\nSource: {source}\nReference: {reference}\nFile Name: {uploaded_file.name}, File Type: {uploaded_file.type}"
        with open(os.path.join(dir_path, 'info.txt'), 'w') as f:
            f.write(message_details)
        slack_payload = {"text": message_details}
        try:
            response = requests.post(slack_webhook_url, json=slack_payload)
            if response.status_code == 200:
                st.success("Your CIF file has been uploaded successfully!")
            else:
                st.error(f"Failed to send to Slack. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

st.info("""
                **Thank you** for your contribution to CoRE MOF database.
                """
                )
