#!/usr/bin/env python
import sys
import warnings
import gradio as gr
from datetime import datetime

from find_candidate.crew import FindCandidate

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

inputs = {
    "job_description": "A frontend developer is responsible for designing and developing the user interface (UI) of websites and web applications, using technologies like HTML, CSS, and JavaScript, to create the visual elements and interactive features that users see and interact with on a website or app; they collaborate with designers, backend developers, and product managers to bring a website's front-end vision to life.",
    "folder_path": "./CV",
}

def run():
    """
    Run the crew.
    """
    
    try:
        FindCandidate(folder_path=inputs['folder_path']).crew(job_description=inputs['job_description']).kickoff(inputs=inputs)
        # FindCandidate().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
    
iface = gr.Interface(
    fn=lambda job_description, folder_path: (
        "Job description and folder path are required." if not job_description or not folder_path 
        else FindCandidate(folder_path=folder_path).crew(job_description=job_description).kickoff(inputs={"job_description": job_description, "folder_path": folder_path})
    ),
    inputs=[
        gr.Textbox(label="Job Description", value=inputs["job_description"]),
        gr.Textbox(label="Folder Path", value=inputs["folder_path"])
    ],
    outputs="text",
    title="CrewAI CV Analyzer",
    description="Upload a CV to extract project details from the CV and write a blog post about the technologies.",
    css="footer{display:none !important}",
    flagging_options=[],
)

iface.launch()