from urllib import response

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()

def main():
        
    raw_text = "Morocco,[c] officially the Kingdom of Morocco,[d] is a country in the Maghreb region of North Africa. It has coastlines on the Mediterranean Sea to the north and the Atlantic Ocean to the west, and has land borders with Algeria to the east, and the disputed territory of Western Sahara to the south, occupied by Morocco since 1975. Morocco also claims the Spanish exclaves of Ceuta, Melilla and Peñón de Vélez de la Gomera, and several small Spanish-controlled islands off its coast.[16] Morocco also claims to share a border with Mauritania through the disputed territory of Western Sahara. It has a population of approximately 37 million. Islam is both the official and predominant religion, while Arabic and Berber are the official languages. Additionally, French and the Moroccan dialect of Arabic are widely spoken. The culture of Morocco is a mix of Arab, Berber, European (specifically Andalusian[17]), and African cultures. Its capital is Rabat, while its largest city is Casablanca.[18]"
    
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
