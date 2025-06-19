# AI-Powered Resume Optimization Agent

This project is a Streamlit web application that leverages the power of Large Language Models (LLMs) through LangChain and the Google Gemini API to help users optimize their resumes for specific job descriptions.

The application implements a multi-step AI agent that analyzes both the job description and the user's resume, provides actionable feedback, and automatically generates a new, tailored resume in both plain text and professional LaTeX formats.

## Demo

## ✨ Features

-   **Smart Document Parsing**: Upload job descriptions and resumes in various formats (`.pdf`, `.docx`, `.txt`).
-   **In-Depth Job Analysis**: Automatically extracts key technical skills, soft skills, experience requirements, and primary responsibilities from the job description.
-   **ATS Compatibility Check**: Evaluates the original resume for Applicant Tracking System (ATS) readability, keyword alignment, and formatting best practices.
-   **Actionable Suggestions**: Generates targeted, expert-level advice on how to bridge the gap between the resume and the job requirements.
-   **Automated Resume Generation**: Creates a new, optimized resume in plain text that incorporates the suggested improvements.
-   **Professional LaTeX Output**: Converts the optimized resume text into a clean, well-structured LaTeX document for a polished, professional look.
-   **Secure & Private**: Your API key and uploaded documents are processed only for the duration of the session and are not stored.

## 🤖 How It Works

The core of this application is a **LangChain Agent** powered by the **Google Gemini Pro** model. The agent is equipped with a suite of custom tools and follows a precise, pre-defined chain of thought to ensure a high-quality output.

The agent executes the following steps in sequence:
1.  **Analyze Job Requirements**: The agent uses the `analyze_job_requirements` tool to break down the job description text into a structured analysis of skills and responsibilities.
2.  **Evaluate Resume**: It then uses the `evaluate_resume_ats_compatibility` tool to assess the user's original resume for strengths, weaknesses, and ATS red flags.
3.  **Suggest Improvements**: With the context from the first two steps, the `suggest_improvements` tool is called to generate specific advice for tailoring the resume.
4.  **Create Optimized Content**: The `create_optimized_resume_content` tool takes the original resume, the job analysis, and the improvement suggestions to write a completely new and optimized resume text.
5.  **Convert to LaTeX**: Finally, the `convert_text_to_latex` tool transforms the optimized text into a professional, compilable LaTeX document.

The final results from all steps are then neatly presented in the Streamlit user interface.

## 🛠️ Technology Stack

-   **Backend & App Framework**:
    -   [Streamlit](https://streamlit.io/): For creating the interactive web application.
    -   [Python 3.9+](https://www.python.org/)
-   **AI & Language Models**:
    -   [LangChain](https://www.langchain.com/): As the primary framework for building the agent and managing LLM interactions.
    -   [Google Gemini API](https://ai.google.dev/): The underlying LLM for all analysis and generation tasks.
-   **Document Processing**:
    -   [PyPDF2](https://pypi.org/project/PyPDF2/): For extracting text from PDF files.
    -   [python-docx](https://pypi.org/project/python-docx/): For extracting text from DOCX files.
-   **Output Format**:
    -   [LaTeX](https://www.latex-project.org/): For creating professionally typeset resume documents.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   Python 3.9 or higher
-   `pip` package manager
-   A [Google API Key](https://ai.google.dev/pricing) with access to the Gemini models.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Dhruv-Chamaria/resumeats.git](https://github.com/Dhruv-Chamaria/resumeats.git)
    cd resume-optimization-agent
    ```

2.  **Create and activate a virtual environment (recommended):**
    -   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    -   **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Create a `requirements.txt` file:**
    Create a file named `requirements.txt` and add the following dependencies to it:

    ```txt
    streamlit
    langchain
    langchain-core
    langchain-google-genai
    google-generativeai
    PyPDF2
    python-docx
    ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The application requires your Google API key to function. You can enter it directly into the web interface when you run the app.

For a more permanent setup (especially for deployment), you can use Streamlit's secrets management:
1.  Create a folder: `.streamlit`
2.  Inside it, create a file: `secrets.toml`
3.  Add your API key to the file:
    ```toml
    GOOGLE_API_KEY = "your_google_api_key_here"
    ```
    The application logic will need to be slightly adjusted to read from `st.secrets` instead of the text input if you want to rely on this method exclusively.

### Running the Application

1.  Make sure your virtual environment is activated.
2.  Run the Streamlit app from your terminal (assuming your script is named `app.py`):
    ```bash
    streamlit run app.py
    ```
3.  The application will open in a new tab in your web browser.

## 🎈 Usage

1.  Navigate to the running Streamlit application in your browser.
2.  Enter your **Google API Key** in the sidebar.
3.  Upload the **Job Description** file (`.pdf`, `.docx`, or `.txt`).
4.  Upload **Your Resume** file (`.pdf`, `.docx`, or `.txt`).
5.  Click the **"🚀 Generate Optimized Resume & Analysis"** button.
6.  Wait for the agent to complete all the steps. The process may take a few moments.
7.  Review the detailed analysis, suggestions, and the final optimized resume text and LaTeX code.
8.  Use the download buttons to save your new resume.