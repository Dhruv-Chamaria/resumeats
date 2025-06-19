import streamlit as st
# The 'openai' import is no longer needed.
# import openai 
from langchain_core.prompts import PromptTemplate
import PyPDF2
import io
import json
from docx import Document
from typing import List, Union, Tuple

# Langchain Imports
from langchain_core.tools import tool
# Updated agent creation to be model-agnostic
from langchain.agents import AgentExecutor, create_tool_calling_agent 
# Swapped out OpenAI for Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Text Extraction Utilities ---
# (No changes needed in this section)

def extract_text_from_pdf(file_bytes: io.BytesIO) -> str:
    """Extracts text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file_bytes)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(file_bytes: io.BytesIO) -> str:
    """Extracts text from a DOCX file."""
    try:
        doc = Document(file_bytes)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""


class ResumeOptimizationAgent:
    """
    A class that encapsulates the tools for the resume optimization agent.
    """
    def __init__(self, api_key: str):
        # --- MODIFIED: Switched from ChatOpenAI to ChatGoogleGenerativeAI ---
        self.llm = ChatGoogleGenerativeAI(
            temperature=0.7,
            google_api_key=api_key,
            model="gemini-2.5-flash" # Using a powerful Gemini model
        )

    @tool
    def analyze_job_requirements(self, jd_text: str) -> str:
        """
        Analyzes the job description (jd_text) to extract key requirements.
        Use this tool to understand the role's needs, such as technical skills,
        soft skills, experience level, and educational background.
        The input is the full text of the job description.
        """
        
        prompt = PromptTemplate.from_template(template = """As a top-tier technical recruiter and ATS architect, analyze this job description meticulously.
        Extract and structure the following information in clear, organized Markdown:
        1.  **Core Technical Skills:** List the essential programming languages, frameworks, and technologies.
        2.  **Key Soft Skills:** Identify crucial non-technical abilities like communication, teamwork, etc.
        3.  **Experience Level & Qualifications:** Determine the required years of experience and educational background.
        4.  **Primary Responsibilities:** Summarize the main duties and expectations of the role.

        Job Description:
        {jd_text}""",
        input_variables=['jd_text']
        )
        print(jd_text)
        print(prompt)   
        # response = self.llm.invoke(prompt)
        # print(response)
        # return response['output']

    @tool
    def evaluate_resume_ats_compatibility(self, resume_text: str) -> str:
        """
        Evaluates the provided resume_text for ATS compatibility, keyword alignment, formatting, and overall effectiveness.
        Use this to identify strengths and weaknesses of the original resume.
        The input is the full text of the candidate's resume.
        """
        prompt = PromptTemplate.from_template(template ="""As an ATS (Applicant Tracking System) expert, evaluate this resume based on the following criteria.
        Provide your analysis in clear, organized Markdown.
        1.  **ATS Parsing & Formatting:** Assess if the layout is clean and machine-readable. Note any complex tables, columns, or graphics that might cause parsing errors.
        2.  **Keyword Optimization:** Check for the presence of relevant industry and role-specific keywords. Is it generic or tailored?
        3.  **Clarity and Impact:** Is the language clear and concise? Are accomplishments quantified with metrics (e.g., 'Increased sales by 20%')?
        4.  **Red Flags & Missing Elements:** Identify any potential red flags (e.g., grammar errors, vague descriptions) or missing sections (e.g., a summary, skills section).

        Resume Text:
        {resume_text}""",
        input_variables=["resume_text"]
        )
        response = self.llm.invoke(prompt)
        return response['output']

    @tool
    def suggest_improvements(self, jd_analysis: str, resume_analysis: str) -> str:
        """
        Provides specific, actionable suggestions to improve the resume based on the job analysis and the resume evaluation.
        Use this tool after analyzing the job description and the resume to bridge the gap between them.
        Inputs are the string outputs from `analyze_job_requirements` and `evaluate_resume_ats_compatibility`.
        """
        prompt = PromptTemplate.from_template(template ="""You are a professional resume writer and career coach with 20 years of experience.
        Based on the provided Job Analysis and Resume Analysis, generate a list of specific, actionable suggestions for improvement.
        Structure your advice in Markdown, focusing on:
        1.  **Keyword Tailoring:** Suggest specific keywords from the job description to integrate into the resume.
        2.  **Impactful Bullet Points:** Provide examples of how to rewrite vague statements into quantifiable, achievement-oriented bullet points.
        3.  **Structural Changes:** Recommend adding, removing, or reordering sections to better align with the job's priorities.
        4.  **Professional Summary:** Advise on creating or refining a summary that directly addresses the employer's needs as outlined in the job analysis.

        Job Analysis:
        {jd_analysis}

        Current Resume Analysis:
        {resume_analysis}""",
        input_variables=["jd_analysis", "resume_analysis"]
        )
        response = self.llm.invoke(prompt)
        return response['output']

    @tool
    def create_optimized_resume_content(self, original_resume_text: str, jd_analysis: str, improvement_suggestions: str) -> str:
        """
        Creates a new, optimized resume in plain text format by applying improvement suggestions
        to the original resume, guided by the job analysis.
        Use this as the final step to generate the optimized resume text.
        Inputs are the original resume text, the job analysis, and the improvement suggestions.
        """
        prompt = PromptTemplate.from_template(template ="""As a senior recruiter and expert resume writer, rewrite and optimize the original resume based on the detailed analysis and suggestions provided.
        Your goal is to create a new resume that is perfectly tailored to the job description and optimized for ATS.

        Job Analysis:
        {jd_analysis}

        Improvement Suggestions:
        {improvement_suggestions}

        Original Resume:
        {original_resume_text}

        **Instructions:**
        1.  Rewrite the resume content to incorporate all the improvement suggestions.
        2.  Ensure the new resume directly reflects the key skills and responsibilities from the job analysis.
        3.  Use strong action verbs and quantify achievements wherever possible.
        4.  Maintain a professional, clean, and easily scannable format.
        5.  The output should be ONLY the full text of the new, optimized resume. Do not include any commentary or extra headers.
        """,
        input_variables=["jd_analysis", "improvement_suggestions", "original_resume_text"]
        )
        response = self.llm.invoke(prompt)
        return response['output']

    @tool
    def convert_text_to_latex(self, optimized_resume_text: str) -> str:
        """
        Converts the given optimized resume text into a professional, well-structured LaTeX document.
        Use this tool on the final optimized resume text to create a high-quality, typeset version.
        Input is the string content of the optimized resume.
        """
        prompt = PromptTemplate.from_template(template ="""Convert the following resume text into a complete, compilable LaTeX document.
        - Use the 'article' class with the `geometry` package for 1-inch margins.
        - Use `\section*` for main section titles (e.g., Experience, Education).
        - Use `itemize` environments for bulleted lists.
        - Ensure the final output is a single block of LaTeX code, starting with `\documentclass{{article}}` and ending with `\end{{document}}`.
        - Do not include any explanations or commentary outside of the LaTeX code itself.

        Resume Text:
        {optimized_resume_text}

        LaTeX Output:
        """,
        input_variables=["optimized_resume_text"]
        )
        response = self.llm.invoke(prompt)
        # Clean the response to ensure it's just the LaTeX code
        if "```latex" in response:
            response = response.split("```latex")[1].split("```")[0].strip()
        
        if not response.strip().startswith("\\documentclass"):
            return f"\\documentclass{{article}}\n\\usepackage[margin=1in]{{geometry}}\n\\begin{{document}}\n% LLM failed to produce valid LaTeX. Content below:\n{optimized_resume_text}\n\\end{{document}}"
        return response['output']


AGENT_SYSTEM_PROMPT_TEMPLATE = """You are an expert resume optimization assistant. Your purpose is to help a user tailor their resume to a specific job description.
You have access to a suite of tools to analyze documents, suggest improvements, and generate optimized content.

Follow these steps precisely:
1.  Analyze the provided job description using the `analyze_job_requirements` tool.
2.  Evaluate the user's original resume using the `evaluate_resume_ats_compatibility` tool.
3.  Based on the outputs from the first two steps, generate targeted advice using the `suggest_improvements` tool.
4.  Using all the information gathered (original resume, job analysis, and suggestions), create the final, optimized resume content using the `create_optimized_resume_content` tool.
5.  Take the optimized text from the previous step and convert it into a professional LaTeX document using the `convert_text_to_latex` tool.

After you have successfully executed all five steps and have the results for each, you MUST provide your final answer as a single, clean JSON dictionary object.
The JSON object must not be inside a code block. It must contain the following exact keys, with their values being the string outputs from the corresponding tools:
- "job_analysis"
- "resume_analysis"
- "improvement_suggestions"
- "optimized_resume_text"
- "latex_resume"
"""

def generate_optimized_resume(jd_text: str, resume_text: str, api_key: str):
    """
    Initializes and runs the LangChain agent to perform the resume optimization task.
    """
    tool_provider = ResumeOptimizationAgent(api_key)
    tools = [
        tool_provider.analyze_job_requirements,
        tool_provider.evaluate_resume_ats_compatibility,
        tool_provider.suggest_improvements,
        tool_provider.create_optimized_resume_content,
        tool_provider.convert_text_to_latex,
    ]

    agent_llm = ChatGoogleGenerativeAI(
        temperature=0, 
        google_api_key=api_key, 
        model="gemini-1.5-pro-latest"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT_TEMPLATE),
            ("user", "Job Description:\n{job_description}\n\nOriginal Resume:\n{original_resume}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


    agent = create_tool_calling_agent(agent_llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=st.secrets.get("AGENT_VERBOSE", True))

    response = agent_executor.invoke({"job_description": jd_text,"original_resume": resume_text})


    output_json_str = response['output']

    try:
        results_dict = json.loads(output_json_str)
        expected_keys = ["job_analysis", "resume_analysis", "improvement_suggestions", "optimized_resume_text", "latex_resume"]
        # Validate that all expected keys are present
        for key in expected_keys:
            if key not in results_dict:
                st.warning(f"Agent output was missing the key: '{key}'. Falling back gracefully.")
                results_dict[key] = f"Error: The agent did not provide the '{key}'."
        return results_dict
    except json.JSONDecodeError as e:
        st.error(f"Fatal Error: Could not parse the JSON output from the agent. {e}")
        st.text_area("Raw Agent Output (Error)", output_json_str, height=200)
        # Return a dictionary with error messages
        return {k: "Error parsing the final agent output. See details above." for k in ["job_analysis", "resume_analysis", "improvement_suggestions", "optimized_resume_text", "latex_resume"]}


def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="Resume Optimization Agent", layout="wide")
    st.title("✨ Resume Optimization Agent")
    # --- MODIFIED: Updated the description to be more model-agnostic ---
    st.write("Upload a job description and your resume, and our AI agent will analyze, optimize, and rebuild it to beat the ATS and impress recruiters. Powered by LangChain and Google Gemini.")

    with st.sidebar:
        st.header("🔑 API Key")
        # --- MODIFIED: Changed the prompt to ask for a Google API Key ---
        api_key = st.text_input("Enter your Google API key:", type="password", help="Your API key is required to run the agent.")
        st.markdown("---")
        st.header("📄 Files")
        jd_file = st.file_uploader("1. Upload Job Description", type=["pdf", "docx", "txt"])
        resume_file = st.file_uploader("2. Upload Your Resume", type=["pdf", "docx", "txt"])
        st.markdown("---")
        st.info("Your data is processed for this session only and is not stored.")

    if jd_file and resume_file:
        # Using BytesIO for consistent handling
        jd_bytes_io = io.BytesIO(jd_file.getvalue())
        resume_bytes_io = io.BytesIO(resume_file.getvalue())

        # Decoding text files and handling docx/pdf
        if jd_file.type == "text/plain":
            jd_text = jd_bytes_io.read().decode('utf-8')
        elif jd_file.type == "application/pdf":
            jd_text = extract_text_from_pdf(jd_bytes_io)
        else: # Handles docx
            jd_text = extract_text_from_docx(jd_bytes_io)

        if resume_file.type == "application/pdf":
            
            resume_text = extract_text_from_pdf(resume_bytes_io)
        elif resume_file.type == "text/plain":
            resume_text = resume_bytes_io.read().decode('utf-8')
        else: # Handles docx
            resume_text = extract_text_from_docx(resume_bytes_io)
    
        if st.button("🚀 Generate Optimized Resume & Analysis", disabled=(not api_key)):

            if not jd_text.strip() or not resume_text.strip():
                st.error("Could not extract text from one or both uploaded files. Please check the files and try again.")
                return

            with st.spinner("🤖 Your AI agent is analyzing, optimizing, and converting... This may take a few moments."):
                try:
                    print(jd_text)
                    print(type(jd_text))
                    results = generate_optimized_resume(jd_text, resume_text, api_key)

                    st.success("✨ Analysis, optimization, and LaTeX conversion completed! ✨")

                    # Display results in three columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        with st.expander("📄 Job Requirements Analysis", expanded=True):
                            st.markdown(results.get('job_analysis', "Not available."))
                    with col2:
                        with st.expander("📑 Current Resume Analysis", expanded=True):
                            st.markdown(results.get('resume_analysis', "Not available."))
                    with col3:
                        with st.expander("💡 Improvement Suggestions", expanded=True):
                            st.markdown(results.get('improvement_suggestions', "Not available."))
                    
                    st.markdown("---")

                    # Display Optimized Resume Text and LaTeX
                    text_col, latex_col = st.columns(2)
                    
                    with text_col:
                        st.subheader("🚀 Optimized Resume (Text)")
                        optimized_text = results.get('optimized_resume_text', "Not available.")
                        st.text_area("Final Resume Text", optimized_text, height=600)
                        st.download_button(
                            label="📥 Download Optimized Resume (TXT)",
                            data=optimized_text,
                            file_name="optimized_resume.txt",
                            mime="text/plain"
                        )
                    
                    with latex_col:
                        st.subheader("📜 Optimized Resume (LaTeX)")
                        latex_output = results.get('latex_resume', "Not available.")
                        st.text_area("LaTeX Code", latex_output, height=600)
                        st.download_button(
                            label="📥 Download LaTeX Source (.tex)",
                            data=latex_output,
                            file_name="optimized_resume.tex",
                            mime="application/x-tex"
                        )
                # --- MODIFIED: Changed the error handling from openai.APIError to a more general exception ---
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())
    elif st.button("Generate Optimized Resume & Analysis"):
        st.warning("Please upload both a job description and a resume, and enter your API key.")


if __name__ == "__main__":
    main()