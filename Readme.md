AI-Powered Resume Optimization Agent
This project is a Streamlit web application that leverages the power of Large Language Models (LLMs) through LangChain and the Google Gemini API to help users optimize their resumes for specific job descriptions.

The application implements a multi-step AI agent that analyzes both the job description and the user's resume, provides actionable feedback, and automatically generates a new, tailored resume in both plain text and professional LaTeX formats.

Demo
✨ Features
Smart Document Parsing: Upload job descriptions and resumes in various formats (.pdf, .docx, .txt).
In-Depth Job Analysis: Automatically extracts key technical skills, soft skills, experience requirements, and primary responsibilities from the job description.
ATS Compatibility Check: Evaluates the original resume for Applicant Tracking System (ATS) readability, keyword alignment, and formatting best practices.
Actionable Suggestions: Generates targeted, expert-level advice on how to bridge the gap between the resume and the job requirements.
Automated Resume Generation: Creates a new, optimized resume in plain text that incorporates the suggested improvements.
Professional LaTeX Output: Converts the optimized resume text into a clean, well-structured LaTeX document for a polished, professional look.
Secure & Private: Your API key and uploaded documents are processed only for the duration of the session and are not stored.
🤖 How It Works
The core of this application is a LangChain Agent powered by the Google Gemini Pro model. The agent is equipped with a suite of custom tools and follows a precise, pre-defined chain of thought to ensure a high-quality output.

The agent executes the following steps in sequence:

Analyze Job Requirements: The agent uses the analyze_job_requirements tool to break down the job description text into a structured analysis of skills and responsibilities.
Evaluate Resume: It then uses the evaluate_resume_ats_compatibility tool to assess the user's original resume for strengths, weaknesses, and ATS red flags.
Suggest Improvements: With the context from the first two steps, the suggest_improvements tool is called to generate specific advice for tailoring the resume.
Create Optimized Content: The create_optimized_resume_content tool takes the original resume, the job analysis, and the improvement suggestions to write a completely new and optimized resume text.
Convert to LaTeX: Finally, the convert_text_to_latex tool transforms the optimized text into a professional, compilable LaTeX document.
The final results from all steps are then neatly presented in the Streamlit user interface.

🛠️ Technology Stack
Backend & App Framework:
Streamlit: For creating the interactive web application.
Python 3.9+
AI & Language Models:
LangChain: As the primary framework for building the agent and managing LLM interactions.
Google Gemini API: The underlying LLM for all analysis and generation tasks.
Document Processing:
PyPDF2: For extracting text from PDF files.
python-docx: For extracting text from DOCX files.
Output Format:
LaTeX: For creating professionally typeset resume documents.
🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Python 3.9 or higher
pip package manager
A Google API Key with access to the Gemini models.