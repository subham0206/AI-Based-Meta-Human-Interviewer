import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
import PyPDF2
import json
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

os.environ["OPENAI_API_KEY"] = "sk-proj-PzzET1s6RIRNZ5jQ6NKcT3BlbkFJTQFoi6voweCdFyaxCeZJ"

class ResumeAnalysis(BaseModel):
    technical_skills_match: Dict[str, float] = Field(description="Dictionary of technical skills and match percentage")
    soft_skills_match: Dict[str, float] = Field(description="Dictionary of soft skills and match percentage")
    experience_relevance: float = Field(description="Percentage of relevant experience")
    education_match: float = Field(description="Education requirement match percentage")
    overall_match: float = Field(description="Overall candidate match percentage")
    strengths: List[str] = Field(description="Key strengths of the candidate")
    gaps: List[str] = Field(description="Areas for improvement")
    detailed_feedback: str = Field(description="Detailed analysis and recommendations")

class InterviewQuestions(BaseModel):
    technical_questions: List[Dict[str, str]] = Field(
        description="List of technical questions with their context"
    )
    behavioral_questions: List[Dict[str, str]] = Field(
        description="List of behavioral questions with their context"
    )
    project_questions: List[Dict[str, str]] = Field(
        description="List of questions about past projects and experience"
    )

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def create_analysis_prompt():
    template = """
    You are an expert AI talent recruiter specialized in Generative AI and Machine Learning roles. 
    Analyze the following job description and resume in detail.
    
    Job Description:
    {jd_text}
    
    Resume:
    {resume_text}
    
    Provide a comprehensive analysis focusing on:
    1. Technical Skills Match:
       - Core ML/AI skills
       - Programming languages
       - Frameworks and tools
       - Cloud and infrastructure
    
    2. Soft Skills Match:
       - Communication
       - Leadership
       - Problem-solving
       - Collaboration
    
    3. Experience Analysis:
       - Relevance to role
       - Project complexity
       - Impact and achievements
    
    4. Education Alignment
    
    5. Detailed Strengths and Gaps
    
    Provide your analysis in a structured format that matches the following schema:
    {format_instructions}
    
    Be specific and quantitative in your assessment. Include percentages for matches 
    and provide detailed reasoning for your evaluations.
    """
    
    parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt, parser

def analyze_resume(llm, prompt, parser, jd_text, resume_text):
    """Analyze resume using GPT-4"""
    formatted_prompt = prompt.format(
        jd_text=jd_text,
        resume_text=resume_text
    )
    
    response = llm.predict(formatted_prompt)
    return parser.parse(response)

def create_visualization(analysis_result):
    """Create comprehensive visualizations for the analysis"""

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}],
               [{'type': 'bar', 'colspan': 2}, None]],
        subplot_titles=('Technical Skills', 'Soft Skills', 'Detailed Scores')
    )
    
    tech_skills_avg = sum(analysis_result.technical_skills_match.values()) / len(analysis_result.technical_skills_match)
    fig.add_trace(
        go.Pie(labels=['Match', 'Gap'],
               values=[tech_skills_avg, 100-tech_skills_avg],
               name="Technical Skills"),
        row=1, col=1
    )
    
    soft_skills_avg = sum(analysis_result.soft_skills_match.values()) / len(analysis_result.soft_skills_match)
    fig.add_trace(
        go.Pie(labels=['Match', 'Gap'],
               values=[soft_skills_avg, 100-soft_skills_avg],
               name="Soft Skills"),
        row=1, col=2
    )

    scores = {
        'Technical Skills': tech_skills_avg,
        'Soft Skills': soft_skills_avg,
        'Experience': analysis_result.experience_relevance,
        'Education': analysis_result.education_match,
        'Overall': analysis_result.overall_match
    }
    
    fig.add_trace(
        go.Bar(x=list(scores.keys()),
               y=list(scores.values()),
               name="Detailed Scores"),
        row=2, col=1
    )
    
    fig.update_layout(height=800, showlegend=True)
    return fig

def create_interview_questions_prompt():
    template = """
    You are an expert technical interviewer for AI/ML positions. Based on the following job description and candidate's resume,
    generate relevant interview questions that will help assess the candidate's fit for the role.
    
    Job Description:
    {jd_text}
    
    Candidate's Resume:
    {resume_text}
    
    Key Areas to Focus:
    1. Technical Skills highlighted in their resume
    2. Past projects and their complexity
    3. Behavioral aspects based on role requirements
    
    For each question, provide:
    - The question itself
    - Context for why this question is relevant
    
    Generate a mix of:
    - Technical questions targeting their specific skills
    - Behavioral questions related to their experience
    - Project-specific questions based on their past work
    
    Provide your questions in a structured format that matches the following schema:
    {format_instructions}
    """
    
    parser = PydanticOutputParser(pydantic_object=InterviewQuestions)
    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt, parser

def generate_interview_questions(llm, prompt, parser, jd_text, resume_text):
    """Generate tailored interview questions using GPT-4"""
    formatted_prompt = prompt.format(
        jd_text=jd_text,
        resume_text=resume_text
    )
    
    response = llm.predict(formatted_prompt)
    return parser.parse(response)

def select_top_candidates(results, num_candidates=2):
    """Select top candidates based on overall match and technical skills"""
    ranked_candidates = sorted(
        results,
        key=lambda x: (
            x['analysis'].overall_match,
            sum(x['analysis'].technical_skills_match.values()) / len(x['analysis'].technical_skills_match)
        ),
        reverse=True
    )
    return ranked_candidates[:num_candidates]

def display_interview_questions(questions):
    """Display generated interview questions in an organized manner"""
    st.write("### Technical Questions")
    for i, q in enumerate(questions.technical_questions, 1):
        st.write(f"**Q{i}:** {q['question']}")
        st.info(f"Context: {q['context']}")
        st.write("---")
    
    st.write("### Behavioral Questions")
    for i, q in enumerate(questions.behavioral_questions, 1):
        st.write(f"**Q{i}:** {q['question']}")
        st.info(f"Context: {q['context']}")
        st.write("---")
    
    st.write("### Project Experience Questions")
    for i, q in enumerate(questions.project_questions, 1):
        st.write(f"**Q{i}:** {q['question']}")
        st.info(f"Context: {q['context']}")
        st.write("---")

def extract_email_from_text(text):
    """Extract email address from text using regex"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else None

def send_interview_email(recipient_email, candidate_name):
    """Send interview invitation email with questions"""

    sender_email = "subhamsrivastava.git@gmail.com"

    sender_password = "mirxb cnek zstb awvt"
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"Interview Invitation - Next Steps"
    
    body = f"""
    Dear {candidate_name},
    
    Thank you for your application. We were impressed with your profile and would like to invite you for an interview.
    
    Please reply to this email with your preferred date and time for the interview.
    
    Best regards,
    Hiring Team
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False


def send_interview_invitations(top_candidates):
    """Send interview invitations to top candidates"""
    st.subheader("📧 Sending Interview Invitations")
    
    for candidate in top_candidates:
        with st.expander(f"Email Status: {candidate['filename']}"):
            email = extract_email_from_text(candidate['resume_text'])
            
            if email:
                st.write(f"Found email: {email}")
                success = send_interview_email(
                    email,
                    candidate['filename'].split('.')[0],
                )
                if success:
                    st.success(f"Interview invitation sent to {email}")
                else:
                    st.error(f"Failed to send invitation to {email}")
            else:
                st.error("No email address found in the resume")

def main():

    
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
    )
    
    prompt, parser = create_analysis_prompt()
    interview_prompt, interview_parser = create_interview_questions_prompt()

    st.header("Job Description")
    jd_text = st.text_area("Paste the job description here:", height=200)
    
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF format)",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if st.button("Analyze Resumes") and jd_text and uploaded_files:
        all_results = []
        
        with st.spinner('Analyzing resumes... This may take a few minutes.'):
            for file in uploaded_files:
                st.write(f"Processing: {file.name}")
                
                resume_text = extract_text_from_pdf(file)
                if resume_text:
                    try:
                        analysis = analyze_resume(llm, prompt, parser, jd_text, resume_text)
                        all_results.append({
                            'filename': file.name,
                            'resume_text': resume_text,
                            'analysis': analysis
                        })
                    except Exception as e:
                        st.error(f"Error analyzing {file.name}: {str(e)}")
        
        if all_results:
            st.header("Analysis Results")
            
            comparison_data = []
            for result in all_results:
                analysis = result['analysis']
                comparison_data.append({
                    'Candidate': result['filename'],
                    'Overall Match': f"{analysis.overall_match:.1f}%",
                    'Technical Skills': f"{sum(analysis.technical_skills_match.values()) / len(analysis.technical_skills_match):.1f}%",
                    'Soft Skills': f"{sum(analysis.soft_skills_match.values()) / len(analysis.soft_skills_match):.1f}%",
                    'Experience': f"{analysis.experience_relevance:.1f}%",
                    'Education': f"{analysis.education_match:.1f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.subheader("Candidates Comparison")
            st.dataframe(comparison_df)
            
            for result in all_results:
                with st.expander(f"Detailed Analysis: {result['filename']}"):
                    analysis = result['analysis']
                    
                    fig = create_visualization(analysis)
                    st.plotly_chart(fig)
                    
                    st.subheader("Technical Skills Breakdown")
                    tech_skills_df = pd.DataFrame.from_dict(
                        analysis.technical_skills_match,
                        orient='index',
                        columns=['Match Percentage']
                    )
                    st.dataframe(tech_skills_df)
                    
                    st.subheader("Soft Skills Breakdown")
                    soft_skills_df = pd.DataFrame.from_dict(
                        analysis.soft_skills_match,
                        orient='index',
                        columns=['Match Percentage']
                    )
                    st.dataframe(soft_skills_df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Key Strengths")
                        for strength in analysis.strengths:
                            st.write(f"✓ {strength}")
                    
                    with col2:
                        st.subheader("Areas for Improvement")
                        for gap in analysis.gaps:
                            st.write(f"○ {gap}")
                    
                    st.subheader("Detailed Feedback")
                    st.write(analysis.detailed_feedback)
            
            st.download_button(
                label="Download Detailed Analysis Report",
                data=json.dumps([{
                    'filename': r['filename'],
                    'analysis': r['analysis'].dict()
                } for r in all_results], indent=2),
                file_name="resume_analysis_report.json",
                mime="application/json"
            )

            top_candidates = select_top_candidates(all_results)
            
            st.header("🌟 Top Candidates & Interview Questions")
            
            candidate_tabs = st.tabs([f"Candidate: {candidate['filename']}" for candidate in top_candidates])
            
            for tab, candidate in zip(candidate_tabs, top_candidates):
                with tab:
                    st.subheader("Candidate Analysis")
                    fig = create_visualization(candidate['analysis'])
                    #st.plotly_chart(fig)
                    st.plotly_chart(fig, key=f"plot_{candidate['filename']}")
                    
                    st.subheader("💡 Recommended Interview Questions")
                    with st.spinner("Generating tailored interview questions..."):
                        interview_questions = generate_interview_questions(
                            llm,
                            interview_prompt,
                            interview_parser,
                            jd_text,
                            candidate['resume_text']
                        )
                        display_interview_questions(interview_questions)
            
            full_report = [{
                'filename': r['filename'],
                'analysis': r['analysis'].dict(),
                'interview_questions': generate_interview_questions(
                    llm, interview_prompt, interview_parser, jd_text, r['resume_text']
                ).dict() if r in top_candidates else None
            } for r in all_results]
            
            st.download_button(
                label="Download Complete Analysis Report",
                data=json.dumps(full_report, indent=2),
                file_name="resume_analysis_report.json",
                mime="application/json"
            )

            send_interview_invitations([{
                'filename': candidate['filename'],
                'resume_text': candidate['resume_text']
            } for candidate in top_candidates])



if __name__ == "__main__":
    main()