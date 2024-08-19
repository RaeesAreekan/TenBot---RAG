import argparse

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
import os
from dotenv import load_dotenv


#Part of the retriever method
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# You are Bot helping students find the right college. With the details you have , help the student. You are provided with some context too
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# def main():
#     # Creating a CLI
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text

#     # Preparing the Database

#     embedding_function = OpenAIEmbeddings()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Searching in the Database

#     results = db.similarity_search_with_relevance_scores(query_text,k=3)
#     if len(results) == 0 or results[0][1] < 0.7:
#         print(f"Unable to find matching results.")
#         return
    
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     print(prompt)

#     # Givving this to the LLM

#     model = ChatOpenAI()
#     # response_text = model.predict(prompt)

#     sources = [doc.metadata.get("source", None) for doc, _score in results]

#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)


######## Now , we are using the retriever chain method that we had learnt earlier in the thru github tut ########## 

vector_store = Chroma(persist_directory=CHROMA_PATH,
embedding_function=OpenAIEmbeddings())

# Setting up retriever

retriever = vector_store.as_retriever()

template = """   You are a bot equipped with helping students on finding more about colleges and related branches.
For the given question , you are given the below context:
{context}

answer the following question : {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Initializing the LLM

llm = ChatOpenAI(temperature=0.1)

# Defining the rag chain

final_rag_chain = (
    {"context": lambda x: "\n".join([doc.page_content for doc in retriever.invoke(x["question"])]), "question":lambda x: x['question']}
    |prompt | llm | StrOutputParser()
)

# Writing the final function to call this

def get_answer(question):
    result = final_rag_chain.invoke({"question":question})
    return result




if __name__ == "__main__":
    # main()
    question = "Which all programs are there in Mechanical Engineering?"
    print(get_answer(question))