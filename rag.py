import json
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# loading secrets
with open('secrets.json') as config_file:
    secrets = json.load(config_file)

# setting up environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = secrets["LANGCHAIN_API_KEY"]
os.environ['OPENAI_API_KEY'] = secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "talk-to-scholar"

# indexing the documents
def pdf_to_pages(transcript_path):
    pages = []

    # Iterate through the files in the directory
    for filename in os.listdir(transcript_path):
        file_path = os.path.join(transcript_path, filename)
        if os.path.isfile(file_path):  # Check if it's a file
            loader = PyPDFLoader(file_path)
            for page in loader.load():
                pages.append(page)
    return pages

# splitting documents into chunks
def splitter(pages):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

    # Make splits
    splits = text_splitter.split_documents(pages)
    return splits

# creating a vectorstore
def indexing(db_name):
    persist_directory = f"data/chroma_db/{db_name}"
    transcript_path = f"data/{db_name}"
    if os.path.isdir(persist_directory):
        vectorstore = Chroma(
                            collection = db_name,
                            embedding = OpenAIEmbeddings(),
                            persist_directory = persist_directory)
    else:
        pages = pdf_to_pages(transcript_path=transcript_path)
        splits = splitter(pages)
        vectorstore = Chroma.from_documents(
                                        collection_name=db_name,
                                        documents=splits, 
                                        embedding = OpenAIEmbeddings(),
                                        persist_directory = persist_directory)
    
    return vectorstore

def retrieval_and_generation(question, db_name = "1801"):
    # creating retriever
    vectorstore = indexing(db_name)
    retriever = vectorstore.as_retriever()
    
    # creating prompt template
    prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know.\nQuestion: {question} \nContext: {context} \nAnswer:"""
    prompt__template = PromptTemplate.from_template(prompt)

    # creating LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # creating rag chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt__template
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(question)
    return response

print(retrieval_and_generation("What is the meaning of partial differentiation?"))

    
