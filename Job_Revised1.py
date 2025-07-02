import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import CSVReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent
from typing import Optional
from llama_index.core import PromptTemplate
from llama_index.llms.gemini import Gemini
import base64
import nest_asyncio
import json
from pathlib import Path
nest_asyncio.apply()

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)

system_prompt = """
You are a multi-lingual career advisor expert who has knowledge based on 
real-time data and document knowledge base. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.
You primary job is to help students find jobs related to their interests from the Jobstreet Platform.
When users ask general career questions, interview tips, or career guidance, use your knowledge base.
For specific job searches, use the JobStreet search tool.
"""

react_system_header_str = """\
## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.
You have access to the following tools:
{tool_desc}
## Output Format
To answer the question, please use the following format.
```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```
Please ALWAYS start with a Thought.
Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.
If this format is used, the user will respond in the following format:
```
Observation: tool response
```
You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:
```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```
```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```
## Additional Rules
- You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.
## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.
"""
react_system_prompt = PromptTemplate(react_system_header_str)

import sys
import logging
import requests

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create a .streamlit/secrets.toml file with your GEMINI_API_KEY.")
    st.stop()

Settings.llm = Gemini(
    model="models/gemini-2.0-flash",
    api_key=GEMINI_API_KEY,
    system_prompt=system_prompt,
    temperature=0
)

Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")

# RAG Index initialization
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    """Load and index documents from the docs folder"""
    docs_path = Path("./docs")
    
    if not docs_path.exists():
        st.warning("Folder 'docs' tidak ditemukan. Membuat folder kosong...")
        docs_path.mkdir(exist_ok=True)
        return None
    
    # Check if docs folder has any files
    if not any(docs_path.iterdir()):
        st.info("Folder 'docs' kosong. RAG knowledge base tidak tersedia.")
        return None
    
    with st.spinner(text="Loading knowledge base – hang tight! This should take a few minutes."):
        try:
            # Read & load document from folder
            reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
            documents = reader.load_data()
            
            if not documents:
                st.warning("Tidak ada dokumen yang berhasil dimuat dari folder 'docs'.")
                return None
            
            # Create vector index
            index = VectorStoreIndex.from_documents(documents, node_parser=splitter)
            st.success(f"Knowledge base berhasil dimuat dengan {len(documents)} dokumen!")
            return index
            
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")
            return None

# Load the knowledge base
knowledge_index = load_knowledge_base()

# Custom Chat Engine Class that combines RAG and ReAct Agent
class RAGReActChatEngine:
    def __init__(self, knowledge_index, tools, llm, memory):
        self.knowledge_index = knowledge_index
        self.tools = tools
        self.llm = llm
        self.memory = memory
        
        # Create RAG chat engine if knowledge base is available
        if knowledge_index is not None:
            self.rag_chat_engine = CondensePlusContextChatEngine(
                verbose=True,
                memory=memory,
                retriever=knowledge_index.as_retriever(similarity_top_k=3),
                llm=llm
            )
        else:
            self.rag_chat_engine = None
            
        # Create ReAct agent for tool calling
        self.react_agent = ReActAgent.from_tools(
            tools=tools,
            verbose=True,
            memory=memory,
            llm=llm,
            system_prompt=react_system_prompt
        )
    
    def _is_job_search_query(self, message):
        """Determine if query is about job search"""
        job_keywords = ['lowongan', 'kerja', 'pekerjaan', 'job', 'vacancy', 'karir', 'cari kerja']
        search_keywords = ['berikan','cari', 'carikan', 'temukan', 'search', 'find']
        
        message_lower = message.lower()
        has_job_keyword = any(keyword in message_lower for keyword in job_keywords)
        has_search_keyword = any(keyword in message_lower for keyword in search_keywords)
        
        return has_job_keyword and has_search_keyword
    
    def chat(self, message):
        """Main chat method that decides whether to use RAG or ReAct agent"""
        try:
            # If it's clearly a job search query, use ReAct agent
            if self._is_job_search_query(message):
                return self.react_agent.chat(message)
            
            # If RAG is available, try RAG first for general questions
            if self.rag_chat_engine is not None:
                try:
                    rag_response = self.rag_chat_engine.chat(message)
                    
                    # If RAG response seems insufficient (too short or generic), fallback to ReAct
                    if len(rag_response.response) < 50 or "don't know" in rag_response.response.lower():
                        return self.react_agent.chat(message)
                    
                    return rag_response
                except Exception as e:
                    st.warning(f"RAG engine error: {e}, falling back to ReAct agent")
                    return self.react_agent.chat(message)
            
            # Fallback to ReAct agent
            return self.react_agent.chat(message)
            
        except Exception as e:
            st.error(f"Chat engine error: {e}")
            return type('Response', (), {'response': 'Maaf, terjadi kesalahan. Silakan coba lagi.'})()
    
    def stream_chat(self, message):
        """Stream chat method"""
        try:
            # For job search, always use ReAct agent
            if self._is_job_search_query(message):
                return self.react_agent.stream_chat(message)
            
            # For general questions, try RAG first if available
            if self.rag_chat_engine is not None:
                try:
                    return self.rag_chat_engine.stream_chat(message)
                except Exception as e:
                    st.warning(f"RAG streaming error: {e}, falling back to ReAct agent")
                    return self.react_agent.stream_chat(message)
            
            # Fallback to ReAct agent
            return self.react_agent.stream_chat(message)
            
        except Exception as e:
            st.error(f"Stream chat error: {e}")
            # Return a simple generator for error case
            def error_gen():
                yield "Maaf, terjadi kesalahan. Silakan coba lagi."
            return type('StreamResponse', (), {'response_gen': error_gen()})()

st.markdown("""
<style>
/* Main container for the header */
.header-container {
    display: flex; /* Use flexbox for alignment */
    align-items: center; /* Vertically center items */
    justify-content: space-between; /* Push items to the ends */
    padding: 10px 0;
    border-bottom: 2px solid #f0f2f6; /* Optional: adds a nice separator line */
    margin-bottom: 20px; /* Space below the header */
}

/* Container for the title and subtitle */
.title-container h1 {
    font-size: 2.5rem; /* Mimic st.title font size */
    font-weight: 600;
    padding: 0;
    margin: 0;
}
.title-container p {
    font-size: 1rem;
    color: #888;
    padding: 0;
    margin: 0;
}

/* Style for the circular logo in the header */
.header-logo {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid #f0f2f6;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>

<div class="header-container">
    <div class="title-container">
        <h1>Student Job-Link</h1>
        <p>powered by jobstreet + RAG knowledge base</p>
    </div>
    <img src="https://cdn-1.webcatalog.io/catalog/jobstreet/jobstreet-icon-filled-256.png?v=1714774884563" class="header-logo">
</div>
""", unsafe_allow_html=True)

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo! Saya adalah asisten karir yang dapat membantu Anda mencari pekerjaan di JobStreet dan memberikan panduan karir. Anda bisa coba salah satu contoh di bawah ini."}
    ]

def job_json_to_natural_language(json_data):
    """Converts the given JSON data into natural language for multiple job listings, including a URL."""
    if "data" not in json_data or not json_data["data"]:
        return "Maaf, tidak ada lowongan yang ditemukan untuk kata kunci tersebut."

    output = ""
    for job in json_data["data"]:
        advertiser_description = job.get("advertiser", {}).get("description", "Nama perusahaan tidak tersedia")
        company_name = job.get("companyName", advertiser_description)
        title = job.get("title", "Judul tidak tersedia")
        locations = ", ".join([loc.get("label", "") for loc in job.get("locations", [])])
        listing_date_display = job.get("listingDateDisplay", "Tanggal tidak tersedia")
        teaser = job.get("teaser", "Deskripsi singkat tidak tersedia")
        work_types = ", ".join(job.get("workTypes", []))
        work_arrangements = ", ".join([wa.get("label", {}).get("text", "") for wa in job.get("workArrangements", {}).get("data", [])])
        
        classification_description = "Klasifikasi tidak tersedia"
        subclassification_description = ""
        if job.get("classifications"):
            classification_description = job["classifications"][0].get("classification", {}).get("description", "Klasifikasi tidak tersedia")
            subclassification_description = job["classifications"][0].get("subclassification", {}).get("description", "")
        
        # Get the job ID to construct the URL
        job_id = job.get("id")
        job_url = f"https://www.jobstreet.co.id/id/job/{job_id}" if job_id else "#"

        job_description = f"""
Lowongan pekerjaan dipublikasikan oleh {advertiser_description} ({company_name}).
Posisi yang ditawarkan adalah {title} di {locations}.
Lowongan ini dipublikasikan {listing_date_display}.
Deskripsi singkat pekerjaan: {teaser}.
Tipe pekerjaan: {work_types}.
Penempatan kerja: {work_arrangements}.
Klasifikasi pekerjaan: {classification_description} {f"- {subclassification_description}" if subclassification_description else ""}.
Url Lowongan: {job_url}
"""
        print(job_description)
        output += job_description + "\n---\n"
    return output.rstrip("\n---\n")

# Declare Tools - hanya JobStreet search tool
async def search_jobstreet(keyword: str) -> str:
    """Searches the JobStreet database for matching entries. Keyword should be words relevant to the job the use is seeking. Try running the function multiple times with different keywords if necessary."""
    try:
        r = requests.get("https://id.jobstreet.com/api/jobsearch/v5/search", params={
            "siteKey": "ID-Main",
            "sourcesystem": "houston",
            "page": "1",
            "worktype": "242",
            "sortmode": "ListedDate",
            "pageSize": "5",
            "include": "seodata,joracrosslink,gptTargeting,pills",
            "locale": "id-ID",
            "keywords": keyword,
            "baseKeywords": keyword
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        return f"# Job Search Results for '{keyword}'\n{job_json_to_natural_language(data)}"
    except requests.RequestException as e:
        return f"An error occurred while searching for jobs: {e}"

# Create tools - hanya satu tool seperti sebelumnya
search_jobstreet_tool = FunctionTool.from_defaults(async_fn=search_jobstreet)
tools = [search_jobstreet_tool]

# Initialize the hybrid chat engine
if "chat_engine" not in st.session_state.keys():
    memory = ChatMemoryBuffer.from_defaults(token_limit=32768)
    st.session_state.chat_engine = RAGReActChatEngine(
        knowledge_index=knowledge_index,
        tools=tools,
        llm=Settings.llm,
        memory=memory
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------- EXAMPLE PROMPTS & JAVASCRIPT INJECTION -----------------
st.markdown("---")
st.markdown("💡 **Try an example prompt:**")

example_prompts = [
    {"title": "Cari Lowongan IT", "prompt": "Carikan saya lowongan pekerjaan di bidang IT atau software engineering"},
    {"title": "Tips Karir", "prompt": "Berikan saya tips untuk memulai karir di bidang teknologi"},
    {"title": "Skill yang Dibutuhkan", "prompt": "Apa skill yang paling dibutuhkan untuk fresh graduate di industri teknologi?"},
    {"title": "Persiapan Interview", "prompt": "Bagaimana cara mempersiapkan diri untuk interview kerja?"}
]

def set_prompt_for_js_injection(prompt_text):
    """This callback sets the prompt text in session state, which will trigger the JS injection."""
    st.session_state.prompt_to_inject = prompt_text

cols = st.columns(2)
for i, p in enumerate(example_prompts):
    with cols[i % 2]:
        st.button(
            p["title"], 
            key=f"prompt_btn_{i}", 
            on_click=set_prompt_for_js_injection, 
            args=[p["prompt"]],
            use_container_width=True
        )

# This block checks if a button was clicked and injects the JS to fill the chat input
if "prompt_to_inject" in st.session_state:
    # Use json.dumps to safely escape the string for JavaScript
    prompt_text_json = json.dumps(st.session_state.prompt_to_inject)

    js_code = f"""
    <script>
        function setChatInputValue(text) {{
            const chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            
            if (chatInput) {{
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(chatInput, text);
                const event = new Event('input', {{ bubbles: true}});
                chatInput.dispatchEvent(event);
            }} else {{
                console.error("Chat input not found.");
            }}
        }}
        setChatInputValue({prompt_text_json});
    </script>
    """
    st.components.v1.html(js_code, height=0)

    # Clean up the session state variable
    del st.session_state.prompt_to_inject

# Display knowledge base status
if knowledge_index is not None:
    st.sidebar.success("✅ Knowledge Base: Aktif")
    st.sidebar.info("RAG akan otomatis digunakan untuk pertanyaan umum tentang karir")
else:
    st.sidebar.warning("⚠️ Knowledge Base: Tidak Aktif")
    st.sidebar.info("Letakkan dokumen di folder 'docs' untuk mengaktifkan RAG")

# ----------------- CHAT INPUT & LOGIC -----------------
if prompt := st.chat_input("Type your message here..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            full_response = st.write_stream(response_stream.response_gen)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Rerun to clear the input box visually and update the display
    st.rerun()