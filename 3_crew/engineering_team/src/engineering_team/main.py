#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime

from engineering_team.crew import EngineeringTeam

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

requirements = """
A simple chat interface for a language learning platform.
The system allows to send either messages or audio recordings.
The system should allow users to record themselves. 
The system should be able to save the recording to a file.
The system should be able to load the recording from a file.
The system should be able to delete all the recordings.
The system shows a button to start recording, and a button to stop recording, The recording is stopped automatically after 10 seconds.
The system should be able to transcribe the recording into text.
The system should be able to play back the recording.
The system returns an echo message whenever the user sends a message.
"""
module_name = "language_learning.py"
class_name = "LanguagesChat"


def run():
    """
    Run the research crew.
    """
    inputs = {
        'requirements': requirements,
        'module_name': module_name,
        'class_name': class_name
    }

    # Create and run the crew
    result = EngineeringTeam().crew().kickoff(inputs=inputs)


if __name__ == "__main__":
    run()