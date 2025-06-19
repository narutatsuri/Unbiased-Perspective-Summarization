import os
import json

import yaml
import boto3
import pickle as pkl
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
from box import Box
from botocore.exceptions import ClientError

from utils import highlight_text, _component


##################
# CONFIGURATIONS #
##################
data_dir_template = "data/data-annotator={}.json"
example_data = json.load(open("data/example_data.json", "r"))
config = Box(yaml.safe_load(open("config.yaml", "r")))

# AWS
aws_access_key_id = ""
aws_secret_access_key = ""
region_name = ""
bucket_name = ""

def upload_data(data_to_upload, upload_name):
    pkl_buffer = BytesIO()
    pkl.dump(data_to_upload, pkl_buffer)
    pkl_buffer.seek(0)

    s3_client.upload_fileobj(pkl_buffer, bucket_name, upload_name)

def download_data(data_dir):
    response = s3_client.get_object(Bucket=bucket_name, Key=data_dir)
    file_content = response['Body'].read()
    
    return pkl.load(BytesIO(file_content))

def next_page():
    if st.session_state.page == 0:
        st.session_state.page += 1
        
        # If user is continuing, fetch old annotations
        try:
            s3_client.head_object(Bucket=bucket_name, Key=data_dir_template.format(st.session_state.name.replace("Annotator ", "")).replace(".json", ".pkl"))
            st.session_state.finished_instances = download_data(data_dir_template.format(st.session_state.name.replace("Annotator ", "")).replace(".json", ".pkl"))

        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                st.session_state.finished_instances = []
                upload_data(st.session_state.finished_instances, data_dir_template.format(st.session_state.name.replace("Annotator ", "")).replace(".json", ".pkl"))
        
        # If user finished all annotations, end session
        if len(st.session_state.finished_instances) == len(st.session_state.data):
            st.success("You have already completed all assigned annotations.")
            st.stop()
    
    elif len(st.session_state.finished_instances) < len(st.session_state.data):
        st.session_state.page += 1

        upload_data(st.session_state.annotations, f"name={st.session_state.name.lower().replace(' ', "-")}.pkl")
        st.session_state.finished_instances.append(st.session_state.current_item)
        upload_data(st.session_state.finished_instances, data_dir_template.format(st.session_state.name.replace("Annotator ", "")).replace(".json", ".pkl"))

    else:
        upload_data(st.session_state.annotations, f"name={st.session_state.name.lower().replace(' ', "-")}.pkl")
        st.session_state.finished_instances.append(st.session_state.current_item)
        upload_data(st.session_state.finished_instances, data_dir_template.format(st.session_state.name.replace("Annotator ", "")).replace(".json", ".pkl"))

        st.success("Survey complete! All annotations have been saved.")
        st.stop()

# Connect to AWS S3 database
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name,
)

st.set_page_config(
    page_title=config.headline,
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="auto",
)

##############################
# CREATING SESSION VARIABLES #
##############################
if "page" not in st.session_state:
    st.session_state.page = 0

if "name" in st.session_state:    
    try:
        s3_client.head_object(Bucket=bucket_name, Key=f"name={st.session_state.name.lower().replace(' ', "-")}.pkl")
        st.session_state.annotations = download_data(f"name={st.session_state.name.lower().replace(' ', "-")}.pkl")
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            st.session_state.annotations = [{} for _ in range(len(st.session_state.data))]

# Display content based on the current page
if st.session_state.page == 0:
    _, main_page, _ = st.columns(config.layout)
    with main_page:
        st.write(config.title, unsafe_allow_html=True)
        st.write(config.article, unsafe_allow_html=True)

        ######################
        # KEY POINT EXAMPLES #
        ######################
        for key_point_example in example_data["key_point_examples"]:
            with st.container(border=True):
                st.write(f'<strong>Topic: {key_point_example["topic"]}</strong>', unsafe_allow_html=True,)
                st.write(f'<p style="color:black; font-size:15px;">\n{highlight_text(key_point_example["article"], key_point_example["evidence"], config.green_highlight)}</p>', unsafe_allow_html=True)
            st.write(f"<blockquote><strong>Excerpt</strong>: {'... '.join(key_point_example['evidence'])}</blockquote>", unsafe_allow_html=True)
            st.write(key_point_example["explanation"])

        st.write(config.interface_demo, unsafe_allow_html=True)
        
        with st.container(border=True):
            st.write(config.highlight_instructions)
            with st.container(border=True):
                st.write(
                    f"<strong>Topic: {example_data['examples'][0]['topic']}</strong>",
                    unsafe_allow_html=True,
                )

                st.session_state.example_highlight_document = _component(
                    text=example_data["examples"][0]["document"],
                    highlights=[],
                    nhighlights=[],
                    mode="text_highlighter",
                )

            st.write(config.highlight_bullets, unsafe_allow_html=True)
            if st.session_state.example_highlight_document:
                for highlight in st.session_state.example_highlight_document[0]:
                    string = "… ".join(
                        [
                            selection["label"].replace("\n", " ").replace("$", "\\$")
                            for selection in highlight
                        ]
                    )

                    if string != "":
                        st.write(f"- {string}")
        st.write(config.annotation_instruction, unsafe_allow_html=True)

        for index, example in enumerate(example_data["examples"]):
            st.write(f"### Example Annotation {index+1 if len(example_data['examples']) > 1 else ""}")
            with st.container(border=True):
                text = "".join(f'<p style="color:black; font-size:12px;">'
                                f'{highlight_text(highlight_text(article, example["evidence"], config.green_highlight), example["no_evidence"], config.red_highlight)}'
                                f"</p>" for article in example["document"].split("\n\n"))
                st.write(text, unsafe_allow_html=True)
            st.write(example["annotation"], unsafe_allow_html=True)
        st.write(config.finishing, unsafe_allow_html=True)

        # Annotator ID Selection
        num_annotators = len([filename for filename in os.listdir("data") if "annotator=" in filename])
        st.radio("Please select your annotator ID from below:", ["Select Option..."] + [f"Annotator {i+1}" for i in range(num_annotators)], key="annotator_name")
        
        if not st.session_state.annotator_name or st.session_state.annotator_name == "Select Option...":
            st.info(f"Please select your Annotator ID.")
        else:
            st.info(f"You have selected your annotator ID to be: {st.session_state.annotator_name}. If this is correct, please continue to the next page.")
            st.session_state.name = st.session_state.annotator_name
            st.session_state.data = json.load(open(data_dir_template.format(st.session_state.name.replace("Annotator ", "")), "r"))
            
else:
    if len(st.session_state.finished_instances) == len(st.session_state.data):
        st.success("Survey complete! All annotations have been saved.")
        st.stop()
    
    _, main_page, _ = st.columns(config.layout)
    with main_page:
        current_item = st.session_state.data[len(st.session_state.finished_instances)]
        
        # Save to session state for recording purposes 
        st.session_state.current_item = current_item
        
        # Write interface
        st.write(config.title, unsafe_allow_html=True)
        st.write(config.highlight_instructions)

        with st.container(border=True):
            st.write(f"<strong>Topic: {current_item['topic']}</strong>", unsafe_allow_html=True)
            st.session_state.annotations[len(st.session_state.finished_instances)]["highlight_document"] = _component(text=current_item["document"], highlights=[], nhighlights=[], mode="text_highlighter")

        # Display highlighted bullets
        st.write(config.highlight_bullets, unsafe_allow_html=True)
        if st.session_state.annotations[len(st.session_state.finished_instances)][ "highlight_document"]:
            for highlight in st.session_state.annotations[len(st.session_state.finished_instances)]["highlight_document"][0]:
                string = "… ".join([selection["label"].replace("\n", " ").replace("$", "\\$") for selection in highlight])
                if string != "":
                    st.write(f"- {string}")
        st.write("---", unsafe_allow_html=True)

if "name" in st.session_state:
    _, next_button, _ = st.columns(config.layout)
    with next_button:
        st.button("Next", on_click=next_page)
