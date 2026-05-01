from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents import create_agent
from deepagents import create_deep_agent
import urllib.error
import urllib.request
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()



# @tool("add_numbers", return_direct=True)
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text

def main():
    print("Starting agent...")
    # SYSTEM_PROMPT = """You are a literary data assistant.

    # ## Capabilities

    # - `fetch_text_from_url`: loads document text from a URL into the conversation.
    # Do not guess line counts or positions—ground them in tool results from the saved file."""

    SYSTEM_PROMPT = """You are a literary data assistant. You may use the tool `fetch_text_from_url` to load document text. Do not fabricate numbers or line positions — verify them with the tool.

When replying, output exactly one JSON object and nothing else (no commentary, no markdown). The JSON must match this schema precisely:

{
  "gatsby_line_count": integer | null,      // number of lines containing "Gatsby"
  "daisy_first_line": integer | null,       // 1-based line number of first line containing "Daisy"
  "synopsis": string,                       // two-sentence neutral synopsis
  "how_you_computed_counts": string,        // brief explanation of method or why fields are null
  "error": string | null                    // error message if a failure occurred, else null
}

Rules:
- Use null for any numeric value you cannot verify from the fetched text.
- Always base counts and line numbers on the exact text returned by `fetch_text_from_url`.
- When counting lines, treat each newline-separated line as one line and count a line if it contains the substring (case-sensitive).
- If you call `fetch_text_from_url`, include the URL and a short summary of the tool output (e.g., number of lines read) in `how_you_computed_counts`.
- If a runtime or fetch error occurs, set `error` to the error message and set other fields to null.
- Do not print anything other than the single JSON object."""

    content = f"""Project Gutenberg hosts a full plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Answer as much as you can:

1) How many lines in the complete Gutenberg file contain the substring `Gatsby` (count lines, not occurrences within a line, each line ends with a line break).
2) The 1-based line number of the first line in the file that contains `Daisy`.
3) A two-sentence neutral synopsis.

Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was and what the error message was."""
    
    # llm = ChatOllama(model="llama3.2:1b", temperature=0.9)
    tools = [fetch_text_from_url]

    checkpointer = InMemorySaver()
    
    # agent = create_agent(llm, tools, system_prompt="You are a calculator")
    
    model = init_chat_model(
        "llama3.2:1b",
        model_provider="ollama",
        temperature=0.5,
        timeout=600,
        max_tokens=25000,
        streaming=True,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    deep_agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "great-gatsby-lc"}},
    )

    deep_agent_result = deep_agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "great-gatsby-da"}},
    )
    
    print(agent_result["messages"][-1].content_blocks)
    print("\n")
    print(deep_agent_result["messages"][-1].content_blocks)

if __name__ == "__main__":
    main()
