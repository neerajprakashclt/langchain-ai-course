from urllib import response

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()

def main():
        
    raw_text = "Some input text here"
    
    summary_template = """
    given the following information {information}, summarize as
    1. A short {mood} summary
    2. Two interesting facts"""
    
    summary_prompt_template = PromptTemplate.from_template(summary_template)

    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0)
    
    chain = summary_prompt_template | llm
    try:
        response = chain.invoke(
            input={
                "information": raw_text, 
                "mood": "funny and sarcastic"
                }
        )
        print(response.content)

    except Exception as e:        
        print(f"An error occurred: {e}")  
    

if __name__ == "__main__":
    main()
