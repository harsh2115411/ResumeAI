import streamlit as st
import PyPDF2 as pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import numpy as np
import io
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

@st.cache_resource(show_spinner=False)
def calculate_similarity_score(resume_text, jd_text):
    """Calculate ATS similarity score between resume and job description"""
    try:
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        texts = [resume_text, jd_text]
        embedding_vectors = embeddings.embed_documents(texts)
        
        resume_embedding = np.array(embedding_vectors[0])
        jd_embedding = np.array(embedding_vectors[1])
        
        similarity = np.dot(resume_embedding, jd_embedding) / (
            np.linalg.norm(resume_embedding) * np.linalg.norm(jd_embedding)
        )
        return round(similarity * 100, 2)
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0

@st.cache_resource(show_spinner=False)
def process_documents(_resume_text, _jd_text):
    """Process resume and job description into vector store"""
    try:
        # Create documents with clear labels
        documents_text = [
            f"RESUME CONTENT:\n{_resume_text}",
            f"JOB DESCRIPTION CONTENT:\n{_jd_text}"
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=80
        )
        document_chunks = text_splitter.create_documents(documents_text)
        
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        db = FAISS.from_documents(document_chunks, embeddings)
        
        return db
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None

def setup_resume_chatbot(resume_text, jd_text, similarity_score):
    """Setup chatbot for resume analysis"""
    db = process_documents(resume_text, jd_text)
    if not db:
        return None, None
        
    memory = ConversationBufferMemory(return_messages=True)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert resume analyzer and career counselor. 
        
        You have access to a candidate's resume and a job description through the context: {{context}}
        
        Key Information:
        - ATS Similarity Score: {similarity_score}%
        
        You can answer questions about:
        - Resume strengths and weaknesses
        - Missing skills or qualifications
        - How well the candidate matches the job requirements
        - Career advice and recommendations
        - Profile summary and key highlights
        - Specific skills, experience, or achievements
        - ATS score explanation and improvement tips
        
        Always provide specific, actionable insights based on the actual content of the resume and job description.
        Be encouraging but honest in your assessments."""),
        MessagesPlaceholder(variable_name="memory"),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    ret = create_retrieval_chain(retriever, document_chain)

    return ret, memory


def main():
    st.set_page_config(
        page_title="Resume Analyzer", 
        layout="wide", 
        page_icon="📄"
    )
    
    # Header
    st.markdown("""
        <h1 style='text-align:center; color:#2E86AB; margin-bottom: 10px;'>
            🎯 AI Resume Analyzer
        </h1>
        <p style='text-align:center; color:#666; font-size: 18px; margin-bottom: 30px;'>
            Upload your resume, paste job description, and get intelligent insights!
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .user-msg {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        margin-left: 20%;
        border: 1px solid #BBDEFB;
    }
    .bot-msg {
        background-color: #F5F5F5;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        margin-right: 20%;
        border-left: 4px solid #2E86AB;
    }
    .score-display {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .score-number {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    .suggestion-box {
        background-color: #F0F8FF;
        border: 1px solid #2E86AB;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #1565C0;
    }
    .upload-section {
        background-color: #F8F9FA;
        border: 2px dashed #2E86AB;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
    }
    .status-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        
        upload_col1, upload_col2 = st.columns([1, 1])
        
        with upload_col1:
            st.markdown("#### 📄 Upload Resume")
            uploaded_resume = st.file_uploader(
                "Choose your resume file",
                type=['pdf'],
                help="Upload your resume in PDF format",
                label_visibility="collapsed"
            )
            
        with upload_col2:
            st.markdown("#### 📋 Job Description")
            job_description = st.text_area(
                "Paste job description",
                height=150,
                placeholder="Copy and paste the complete job description here...",
                help="Copy and paste the complete job description",
                label_visibility="collapsed"
            )
        
        st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
        process_button = st.button("🔍 Analyze Resume", type="primary", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    if process_button and uploaded_resume and job_description:
        with st.spinner("📄 Processing documents and calculating similarity..."):
            resume_text = extract_text_from_pdf(uploaded_resume)
            
            if resume_text:
                similarity_score = calculate_similarity_score(resume_text, job_description)
                
                chatbot, memory = setup_resume_chatbot(resume_text, job_description, similarity_score)
                
                if chatbot:
                    st.session_state.chatbot = chatbot
                    st.session_state.memory = memory
                    st.session_state.history = []
                    st.session_state.similarity_score = similarity_score
                    st.session_state.resume_text = resume_text
                    st.session_state.jd_text = job_description
                    st.session_state.processed = True
                    
                    st.success("✅ Documents processed successfully! You can now ask questions.")
            else:
                st.error("❌ Could not extract text from the PDF. Please check your file.")

    if st.session_state.get("processed", False):

        score = st.session_state.similarity_score
        st.markdown(f"""
        <div class="score-display">
            <h3>🎯 ATS Compatibility Score</h3>
            <div class="score-number">{score}%</div>
            <p>{"Excellent Match!" if score >= 80 else "Good Match!" if score >= 60 else "Needs Improvement" if score >= 40 else "Poor Match"}</p>
        </div>
        """, unsafe_allow_html=True)
        

        chat_col, status_col = st.columns([2.5, 1])
        
        with chat_col:
            st.markdown("### 💬 Ask Questions About Your Resume")
            
            user_input = st.chat_input("Ask me anything about your resume analysis...")
            
            if user_input:
                st.session_state.history.append(("user", user_input))
                
                with st.spinner("🤖 Analyzing..."):
                    try:
                        response = st.session_state.chatbot.invoke({
                            "input": user_input,
                            "memory": st.session_state.memory.chat_memory.messages
                        })
                        bot_response = response.get("answer", "I apologize, but I couldn't process your question properly.")
                        
                        st.session_state.history.append(("bot", bot_response))
                    
                        st.session_state.memory.chat_memory.add_user_message(user_input)
                        st.session_state.memory.chat_memory.add_ai_message(bot_response)
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

            chat_container = st.container()
            with chat_container:
                for role, message in st.session_state.history:
                    if role == "user":
                        st.markdown(f'<div class="user-msg">👤 {message}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="bot-msg">🤖 {message}</div>', unsafe_allow_html=True)
        
        with status_col:
            st.markdown("""
            <div class="status-card">
                <h4 style='color:#2E86AB; margin-bottom:15px;'>📊 Document Status</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ Documents processed")
            st.info(f"📄 Resume: {len(st.session_state.resume_text)} characters")
            st.info(f"📋 Job Description: {len(st.session_state.jd_text)} characters")
            
            st.markdown("---")
            st.markdown("""
            <div class="suggestion-box">
                <strong>💡 Sample Questions:</strong><br>
                • What are my key strengths?<br>
                • What skills am I missing?<br>
                • How can I improve my ATS score?<br>
                • What experience matches best?<br>
                • Suggest resume improvements
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("""
            <small>
            <strong>🔒 Privacy:</strong> Your documents are processed securely and not stored permanently.
            <br><br>
            <strong>⚡ Powered by:</strong> LangChain + Groq + OpenAI
            </small>
            """, unsafe_allow_html=True)

    else:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        

    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI-powered resume analyzer helps you:
        - Calculate ATS compatibility score
        - Identify missing skills
        - Get personalized career advice
        - Optimize your resume for specific jobs
        """)
        
        st.markdown("### 📝 How to Use")
        st.markdown("""
        1. Upload your resume (PDF) above
        2. Paste the job description in the text area
        3. Click "Analyze Resume" to process
        4. Ask questions about your resume match!
        """)
        
        st.markdown("---")
        st.markdown("""
        **💡 Tip:** Be specific with your questions for better insights!
        """)

if __name__ == "__main__":
    main()