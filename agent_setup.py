# agent_setup.py
# ----------------
# Synapse AI Agent Setup (Streamlit Cloud compatible)
# ----------------

import os
import sys
import datetime
from dotenv import load_dotenv
from supabase.client import create_client

# --- Load environment variables ---
load_dotenv()  # .env optional; Streamlit secrets work too

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

os.environ["GOOGLE_API_KEY"] = GEMINI_KEY  # for LangChain Google API

# --- Initialize Supabase client ---
try:
    SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    print(f"FATAL: Could not initialize Supabase Client: {e}")
    sys.exit(1)

# --- LangChain Imports ---
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent_executor import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ---------------- TOOLS ---------------- #

@tool
def support_faq_solver(query: str) -> str:
    """RAG tool for FAQ retrieval."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_query"
        )
        vector_store = SupabaseVectorStore(
            embedding=embeddings,
            client=SUPABASE_CLIENT,
            table_name="documents",
            query_name="match_documents"
        )
        docs = vector_store.similarity_search(query, k=3)

        if not docs:
            return "No relevant information found in the knowledge base."

        return "\n---\n".join([doc.page_content for doc in docs])

    except Exception as e:
        return f"Knowledge base search failed: {e}"


@tool
def create_support_ticket_supabase(user_query: str) -> str:
    """Escalates complex queries to Supabase."""
    payload = {
        "user_query": user_query,
        "agent_note": "Escalated due to insufficient RAG context.",
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        SUPABASE_CLIENT.table("support_tickets").insert(payload).execute()
        return "Your query has been escalated to a human agent."
    except Exception as e:
        return f"Escalation failed: {e}"

# ---------------- LLM & AGENT ---------------- #

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GEMINI_KEY
)

system_prompt = """You are Synapse AI (Tier-1 Support Agent).

Rules:
1. ALWAYS call support_faq_solver first.
2. If no data found → escalate using create_support_ticket_supabase.
3. Always respond politely.

Knowledge Base for Support:
- Password resets, account info, refunds, escalation rules, profile updates,
  login issues, security alerts, subscription plans, payment problems.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

tools = [support_faq_solver, create_support_ticket_supabase]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("✅ Agent Executor Ready!")
