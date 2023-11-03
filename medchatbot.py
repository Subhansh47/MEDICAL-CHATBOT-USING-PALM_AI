from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

key = st.secrets['GOOGLE_API_KEY']
llm = GooglePalm(google_api_key=key, temperature=.7)

embeddings = HuggingFaceEmbeddings()
vectordb_file_path = "faiss_index"

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever()

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

def ask(query):
    chain=get_qa_chain()
    results = chain(query)
    if results:
        print("You Asked")
        print(results['query'])
        print("\n")
        
        print("Your Answer is")
        print(results['result'])
        print("\n")
        
        
        print("Answer is found from -")
        print(results['source_documents'])
        print("\n")
    else:
        print("No results found.")

if __name__ == "__main__":
    chain = get_qa_chain()
    ask('Hidradenitis suppurativa')
