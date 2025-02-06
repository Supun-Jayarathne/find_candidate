#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from find_candidate.crew import FindCandidate

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    inputs = {
        "job_description": "A frontend developer is responsible for designing and developing the user interface (UI) of websites and web applications, using technologies like HTML, CSS, and JavaScript, to create the visual elements and interactive features that users see and interact with on a website or app; they collaborate with designers, backend developers, and product managers to bring a website's front-end vision to life."
    }
    
    try:
        FindCandidate().crew(job_description=inputs['job_description']).kickoff(inputs=inputs)
        # FindCandidate().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")