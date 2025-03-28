import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re

# Set the tokenizers parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SIUC Graduate School ChatAdvisor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for better chat appearance with SIUC maroon theme and message alignment
st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: #ffffff;
    }
    .user-message {
        background-color: #7a1137;
        border: 1px solid #5a0c28;
        margin-left: 20%;
        margin-right: 2%;
    }
    .assistant-message {
        background-color: #96153f;
        border: 1px solid #7a1137;
        margin-right: 20%;
        margin-left: 2%;
    }
    .message-content {
        margin-top: 0.5rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .message-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .user-header {
        justify-content: flex-end;
    }
    .assistant-header {
        justify-content: flex-start;
    }
    .stApp {
        background-color: #4a0a24;
    }
    /* Make sure all text is visible */
    .st-emotion-cache-1y4p8pa {
        color: white;
    }
    .st-emotion-cache-16idsys p {
        color: white;
    }
    .st-emotion-cache-16idsys {
        color: white;
    }
    /* Style sidebar */
    .st-emotion-cache-1rtdyuf {
        color: white;
    }
    /* Style headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    /* Style markdown text */
    .st-emotion-cache-uf99v8 {
        color: white;
    }
    /* Style chat input */
    .st-emotion-cache-1x8kmo4 {
        background-color: #96153f;
    }
    /* Style for bullet points */
    .message-content br + • {
        display: inline-block;
        margin-left: 1em;
    }
    
    .message-content br + [0-9] {
        display: inline-block;
        margin-left: 1em;
    }
    
    /* Add spacing between list items */
    .message-content br {
        display: block;
        margin: 0.3rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def format_response(text):
    """
    Enhanced formatting function to handle bullets and numbers
    """
    import re
    
    # Add spacing around lists
    text = re.sub(r'(\n[•\-\d.])', r'\n\n\1', text)
    
    # Convert markdown-style bullets to HTML bullets
    text = re.sub(r'\n\s*[-•]\s', '\n<br>• ', text)
    
    # Convert markdown-style numbers to HTML
    text = re.sub(r'\n\s*(\d+)\.\s', r'\n<br>\1. ', text)
    
    # Add proper spacing
    text = text.replace('\n\n', '<br><br>')
    
    return text

# Initialize embeddings and load the pre-built FAISS index
@st.cache_resource
def initialize_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4"
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template="""You are a helpful SIUC Graduate School advisor.
                Use the following pieces of context to answer the question. 
                When responding:
                • Use bullet points (•) for listing items or options
                • Use numbers (1., 2., etc.) for sequential steps or processes
                • Structure responses clearly with appropriate formatting
                • Start new lists on a new line
                • Add a blank line before and after lists

                Context: {context}
                
                Question: {question}
                
                Answer: Let me help you with that.""",
                input_variables=["context", "question"]
            )
        }
    )
    return chain

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main title with SIUC branding
st.title("🎓 SIUC Graduate School ChatAdvisor")
st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This AI assistant helps answer questions about:
    - Graduate programs
    - Admission requirements
    - Application process
    - Financial aid
    - And more!
    """)
    st.markdown("---")
    st.markdown("### Tips")
    st.markdown("""
    - Be specific in your questions
    - Ask one question at a time
    - Provide context when needed
    """)

# Initialize the chain
chain = initialize_chain()

# Chat interface
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-header user-header">
                        <div><strong>You</strong> 👤</div>
                    </div>
                    <div class="message-content">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            formatted_content = format_response(message["content"])
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-header assistant-header">
                        <div>🎓 <strong>SIUC Advisor</strong></div>
                    </div>
                    <div class="message-content">{formatted_content}</div>
                </div>
            """, unsafe_allow_html=True)

# Chat input
user_question = st.chat_input("Ask me anything about SIUC Graduate School...")

if user_question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Get response from chain using .invoke() instead of direct call
    response = chain.invoke({
        "question": user_question, 
        "chat_history": [(m["content"], n["content"]) 
                        for m, n in zip(st.session_state.messages[::2], 
                                      st.session_state.messages[1::2])]
    })
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    
    # Rerun to update the chat display
    st.rerun()

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun() 