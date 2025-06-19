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
            st.write(f"<strong>Example Annotation {index+1 if len(example_data['examples']) > 1 else ""}</strong>", unsafe_allow_html=True)
            with st.container(border=True):
                text = "".join(f'<p style="color:black; font-size:12px;">'
                                f'{highlight_text(highlight_text(article, example["evidence"], config.green_highlight), example["no_evidence"], config.red_highlight)}'
                                f"</p>" for article in example["document"].split("\n\n"))
                st.write(text, unsafe_allow_html=True)
            st.write(example["annotation"], unsafe_allow_html=True)

        ######################
        # SUMMARY ANNOTATION #
        ######################
        st.write(config.identify_summary, unsafe_allow_html=True)

        with st.container(border=True):
            with st.container(border=True):
                st.write(example["summary"])
            
            summary_points = st.text_area("Please start each key point on a new line.")
            summary_points = [line.strip() for line in summary_points.splitlines() if line.strip()]
            st.session_state.example_highlight_summary = summary_points

            with st.container(border=True):
                document_evidence_list, summary_evidence_list = st.columns(2)
                with document_evidence_list:
                    st.write("##### In Article")
                    if st.session_state.example_highlight_document:
                        if len(st.session_state.example_highlight_document[0]) == 1 and st.session_state.example_highlight_document[0] == []:
                            st.write("<span style=\"color:gray\">No evidence selected.</span>", unsafe_allow_html=True)        
                                
                        for highlight in st.session_state.example_highlight_document[0]:
                            string = "… ".join([selection["label"].replace("\n", " ").replace("$", "\\$") for selection in highlight])
                            
                            if string != "":
                                st.write(f"- {string}")

                with summary_evidence_list:
                    st.write("##### In Summary")
                    if len(st.session_state.example_highlight_summary) == 0:
                        st.write("<span style=\"color:gray\">No evidence selected.</span>", unsafe_allow_html=True)
                    else:
                        for highlight in st.session_state.example_highlight_summary: 
                            st.write(f" - {highlight}")
        
        underlined_summary = example["summary"]
        for summary_excerpt in example["summary_evidence"]:
            underlined_summary = underlined_summary.replace(summary_excerpt, f'<span style="background-color: #c8ffc8;">{summary_excerpt}</span>')
        
        st.write(config.checklist.format(underlined_summary), unsafe_allow_html=True)
        
        with st.container(border=True):
            document_evidence_list, summary_evidence_list = st.columns(2)

            with document_evidence_list:
                st.write(f"##### In Article\n\n{'\n'.join([' - ' + i for i in example['evidence']])}", unsafe_allow_html=True)

            with summary_evidence_list:
                st.write(f"##### In Summary\n\n{'\n'.join([' - ' + i for i in example['summary_evidence']])}", unsafe_allow_html=True)

        st.write(config.key_point_matching, unsafe_allow_html=True)
        
        with st.container(border=True):        
            st.write(f"##### {config.in_doc_nin_summary}")
            for opt_index, evidence in enumerate(example['evidence']):
                st.checkbox(evidence, key=f"unchecked_example_in_doc_{opt_index}")

            st.write(f"##### {config.nin_doc_in_summary}")
            for opt_index, evidence in enumerate(example['summary_evidence']):
                st.checkbox(evidence, key=f"unchecked_example_in_summary_{opt_index}")  
                
        st.write(config.key_point_matching_example, unsafe_allow_html=True)
        
        with st.container(border=True):        
            st.write(f"##### {config.in_doc_nin_summary}")
            for opt_index, evidence in enumerate(example['evidence']):
                st.checkbox(evidence, key=f"example_in_doc_{opt_index}", value=example["in_doc_nin_summary"][opt_index])

            st.write(f"##### {config.nin_doc_in_summary}")
            for opt_index, evidence in enumerate(example['summary_evidence']):
                st.checkbox(evidence, key=f"example_in_summary_{opt_index}", value=example["nin_doc_in_summary"][opt_index])   
        
        ##########################
        # Annotator ID Selection #
        ##########################
        st.write(config.finishing, unsafe_allow_html=True)
        
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


        ######################
        # ANNOTATE SUMMARIES #
        ######################
        st.write(config.summary_annotation_title)
        
        for index, method in enumerate(current_item["summaries"]):
            summary = current_item["summaries"][method]
            
            st.write(config.summary_title.format(index+1), unsafe_allow_html=True)

            with st.container(border=True):
                st.write(summary)
            
            # Collect extracted key points
            summary_points = st.text_area("Please start each key point on a new line.", key=f"highlight_summary_{method}")
            summary_points = [line.strip() for line in summary_points.splitlines() if line.strip()]
            st.session_state.annotations[len(st.session_state.finished_instances)][f"highlight_summary_{method}"] = summary_points

            # Display side-by-side key points in document and summary
            with st.container(border=True):
                document_evidence_list, summary_evidence_list = st.columns(2)

                with document_evidence_list:
                    st.write("##### In Article")
                    if st.session_state.annotations[len(st.session_state.finished_instances)]["highlight_document"]:
                        if len(st.session_state.annotations[len(st.session_state.finished_instances)]["highlight_document"][0]) == 1 and st.session_state.annotations[len(st.session_state.finished_instances)]["highlight_document"][0][0] == []:
                            st.write("<span style=\"color:gray\">No evidence selected.</span>", unsafe_allow_html=True)        
                                
                        for highlight in st.session_state.annotations[len(st.session_state.finished_instances)]["highlight_document"][0]:
                            string = "… ".join([selection["label"].replace("\n", " ").replace("$", "\\$") for selection in highlight])
                            
                            if string != "":
                                st.write(f"- {string}")

                with summary_evidence_list:
                    st.write("##### In Summary")
                    if len(st.session_state.annotations[len(st.session_state.finished_instances)][f"highlight_summary_{method}"]) == 0:
                        st.write("<span style=\"color:gray\">No evidence selected.</span>", unsafe_allow_html=True)
                    else:
                        for highlight in st.session_state.annotations[len(st.session_state.finished_instances)][f"highlight_summary_{method}"]:
                            st.write(f"- {highlight}")
            
            # Write checkboxes
            st.write(f"##### {config.in_doc_nin_summary}")
            
            if st.session_state.annotations[len(st.session_state.finished_instances)]["highlight_document"] is None or (len(st.session_state.annotations[len(st.session_state.finished_instances)]["highlight_document"][0]) == 1 and st.session_state.annotations[len(st.session_state.finished_instances)]["highlight_document"][0][0] == []):
                st.write("<span style=\"color:gray\">No evidence selected.</span>", unsafe_allow_html=True)        
            else:
                for opt_index, option in enumerate(st.session_state.annotations[len(st.session_state.finished_instances)][f"highlight_document"][0]):
                    option_str = "… ".join([selection["label"].replace("\n", " ").replace("$", "\\$") for selection in option])
                    if option_str != "":    
                        st.session_state.annotations[len(st.session_state.finished_instances)][f"in_doc_nin_summary_{method}_{opt_index}"] = st.checkbox(option_str, key=f"in_doc_nin_summary_{method}_{opt_index}")
            
            st.write(f"##### {config.nin_doc_in_summary}")
            
            if st.session_state.annotations[len(st.session_state.finished_instances)][f"highlight_summary_{method}"] is None or (len(st.session_state.annotations[len(st.session_state.finished_instances)][f"highlight_summary_{method}"]) == 1 and st.session_state.annotations[len(st.session_state.finished_instances)][f"highlight_summary_{method}"][0] == []):
                st.write("<span style=\"color:gray\">No evidence selected.</span>", unsafe_allow_html=True)        
            else:
                for opt_index, option in enumerate(st.session_state.annotations[len(st.session_state.finished_instances)][f"highlight_summary_{method}"]):
                    st.session_state.annotations[len(st.session_state.finished_instances)][f"nin_doc_in_summary_{method}_{opt_index}"] = st.checkbox(option, key=f"nin_doc_in_summary_{method}_{opt_index}")
            
            st.write("<br>", unsafe_allow_html=True)


if "name" in st.session_state:
    _, next_button, _ = st.columns(config.layout)
    with next_button:
        st.button("Next", on_click=next_page)
