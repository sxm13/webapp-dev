import streamlit as st
import os, requests
import gspread
from datetime import date
from oauth2client.service_account import ServiceAccountCredentials


st.set_page_config(page_title="Report of CoRE MOF 2024 Database", page_icon="./figures/logo.png")

logo_url = "./figures/logo.png"
st.sidebar.image(logo_url)
st.logo(logo_url, link="https://zenodo.org/communities/core-mofs/records?q=&l=list&p=1&s=10&sort=newest")

st.title('Report an Issue')

with st.sidebar:
    st.markdown("© 2025 CoRE MOF Project. CC-BY-4.0 License.")

st.divider()

def create_directory(name):
    date_str = date.today().strftime("%Y/%m/%d")
    base_dir = "data/user/"
    i = 0
    while True:
        dir_path = os.path.join(base_dir, f"{date_str}{name}{i}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            return dir_path
        i += 1


scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    "coremof-report-e277b6b42561.json", scope
)
gc = gspread.authorize(creds)

SPREADSHEET_URL = ""
sheet = gc.open_by_url(SPREADSHEET_URL).sheet1


slack_webhook_url = ""

name = st.text_area("Name", "David Jones and Mary Johnson", height=68, key="name_issue")
email = st.text_area("Email", "abc123@gmail.com", height=68, key="email_issue")
affi = st.text_area("Institution", "School of Chemical Engineering, Pusan National University", height=68, key="affi_issue")
vers = st.text_area("Version", "which version of CoRE MOF DB is used", height=68, key="version_issue")
cif_name = st.text_area("CIF Name", "2020[Cu][sql]2[ASR]1", height=68, key="cif_name")
info = st.text_area("Question or Structure Information", "DOI, issue...", height=150, key="info_issue")
uploaded_file = st.file_uploader("Upload CIF file", type=['cif'], key="upload_cif")
agree = st.checkbox("I agree to share this CIF file.", key="agree_cif")

if st.button("📬 Submit Issue"):
    if not (name and email and affi and info):
        st.error("Please complete all required fields.")
    elif uploaded_file and not agree:
        st.error("You must agree to share the CIF file if uploading.")
    else:
        dir_path = create_directory(name)
        with open(os.path.join(dir_path, 'info.txt'), 'w') as f:
            f.write(f"{name}\t{email}\t{affi}\t{info}\n")
        
        message_details = f"New message received:\nName: {name}\nEmail: {email}\nInstitution: {affi}\nCIF Name: {cif_name}\nVersion of CoRE MOF DB: {vers}\nInformation: {info}"
        
        if uploaded_file:
            file_path = os.path.join(dir_path, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            message_details += f"\nFile Name: {uploaded_file.name}, File Type: {uploaded_file.type}"
        
        slack_payload = {"text": message_details}
        try:
            response = requests.post(slack_webhook_url, json=slack_payload)
            if response.status_code == 200:
                st.success("Your issue has been submitted successfully!")
            else:
                st.error(f"Failed to send to Slack. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

        today = date.today().strftime("%Y/%m/%d")
        row = [
            today,
            cif_name,
            vers,        
            info,        
            name,        
            affi,        
            email       
        ]
       
        sheet.append_row(row, value_input_option="USER_ENTERED")
        st.success("Reported successfully! Thank you.")


st.info("""
                **Thank you** for your contribution to CoRE MOF database.
                """
                )

# @st.dialog("Submit Your Issue or CIF")
# def submit_info():
#     st.header("Please fill in your information below.")
    
#     name = st.text_area("Name", "", height=68)
#     email = st.text_area("Email", "", height=68)
#     affi = st.text_area("Institution", "", height=68)
#     info = st.text_area("Question or Structure information", "DOI, structure name...", height=150)
#     uploaded_file = st.file_uploader("Upload CIF file", type=['cif'])
#     agree = st.checkbox("I agree to share this CIF file.")

#     if st.button('📬 Reporting Errors or Uploading your CIF'):
#         if not (name and email and affi and info):
#             st.error("Please complete all required fields.")
#             return

#         dir_path = create_directory(name)
#         message_details = f"New message received:\nName: {name}\nEmail: {email}\nInstitution: {affi}\nInformation: {info}"
        
#         with open(os.path.join(dir_path, 'info.txt'), 'w') as f:
#             f.write(f"{name}\t{email}\t{affi}\t{info}\n")

#         if uploaded_file and agree:
#             file_path = os.path.join(dir_path, uploaded_file.name)
#             with open(file_path, 'wb') as f:
#                 f.write(uploaded_file.getbuffer())
#             message_details += f"\nFile Name: {uploaded_file.name}, File Type: {uploaded_file.type}"
#         st.success("Your information has been submitted successfully!")
#         slack_payload = {"text": message_details}
#         try:
#             response = requests.post(slack_webhook_url, json=slack_payload)
#             if response.status_code == 200:
#                 st.success("Your information has been submitted successfully!")
#             else:
#                 st.error(f"Failed to send to Slack. Status code: {response.status_code}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"An error occurred: {e}")
# submit_info()




# from streamlit_extras.let_it_rain import rain
# rain(
#         emoji="🧧",
#         font_size=54,
#         falling_speed=5,
#         animation_length=1,
#     )


# st.markdown('''
#             contribution
#             *   2023-11-15  62 MOFs loss counterions in the pores by Dr. Özgür.
#             *   2023-06-01  432 manually cleaned MOFs from [ACS Appl. Mater. Interfaces 2023, 15, 23, 28084-28092](https://pubs.acs.org/doi/full/10.1021/acsami.3c04079).
#             *   2022-12-19  2 MOFs missing hydrogen atoms by Mr. Lin.
#             ''')


# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#                 f"""
#                 <style>
#                 .stApp {{
#                     background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#                     background-size: cover
#                 }}
#                 </style>
#                 """,
#                 unsafe_allow_html=True
#                 )
# add_bg_from_local('./figures/bg.png')


# from streamlit_timeline import timeline
# with open('./data/info/td.json', "r") as f:
#     time_data = f.read()

# timeline(time_data, height=400)