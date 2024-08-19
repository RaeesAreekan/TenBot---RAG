from langchain.chains import RetrievalQA  
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
import pinecone
from dotenv import load_dotenv
import time
# import openai

load_dotenv()

# openai.api_key = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


def get_answer(question:str)->str:

    llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0.0
)

# Initialize a LangChain object for retrieving information from Pinecone.
    knowledge = PineconeVectorStore.from_existing_index(
    index_name="tensors-chat",
    embedding=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
)


    custom_template = """
    You are an AI Assistant made to assist students for finding more about colleges , thier curriculum , fee structure etc . If you think th answer is not enough , 
    convey that your knowldge is limited on that topic. You are a chatbot made by Tensors of IIT Madras , a student run NGO. You are also free to search the Web , if you feel you can get more info 
    Answer to the questions concisely and in about 50- 75 words , including all the important points

    {context}

    Based on the context above, please answer the following question:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(custom_template)

# Initialize a LangChain object for chatting with the LLM
# with knowledge from Pinecone. 

######## I have followed to methods , one in QA_Retriever and the second in basic Rag chain #################
    # qa = RetrievalQA.from_chain_type(
    # llm=llm,
    # chain_type="stuff",
    # retriever=knowledge.as_retriever(),
    # return_source_documents=True,
    # prompt = custom_template
    # )

     
    # query = "Tell me about Electrical Engineering at IIT M?"

    # result = qa.invoke({"question" : question})
    # return result['result']
    chain = ({
        "context" : knowledge.as_retriever() , "question" :RunnablePassthrough()
    }
    | prompt |llm | StrOutputParser()
    
    )
    res = chain.invoke(question)
    return res
    

    # print(f"Question: {query}")
    # print(f"Answer: {result['result']}")
    # print(f"Source Documents: {result['source_documents']}")




if __name__ == "__main__":
    print(get_answer("Who created you?"))






    


# knowledge = PineconeVectorStore.from_existing_index(
#     index_name="tensors-chat",
    
#     embedding=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
# )



# def initialize_pinecone():
#     pinecone.init(api_key=PINECONE_API_KEY)
#     index = pinecone.Index('tensors-chat')
#     time.sleep(1)
#     return index

# def create_retriever(index):
#     embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
#     retriever = PineconeVectorStore(index=index, embedding_function=embed_model.embed_query)
#     return retriever

# def main():
#     index = initialize_pinecone()
#     retriever = create_retriever(index)

#     llm = ChatOpenAI(
#     openai_api_key=os.environ.get("OPENAI_API_KEY"),
#     model_name="gpt-3.5-turbo",
#     temperature=0.1
#     )

#     qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True
#     )
    # query = "What is the potential of Electrical Engineering in Future"

    # result = qa_chain(query)

    # print(f"Question: {query}")
    # print(f"Answer: {result['result']}")
    # print(f"Source Documents: {result['source_documents']}")






