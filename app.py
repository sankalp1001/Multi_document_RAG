import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def extract_pdf_text(pdf_documents):
    combined_text = ""
    for pdf in pdf_documents:
        reader = PdfReader(pdf)
        for page in reader.pages:
            combined_text += page.extract_text()
    return combined_text


def split_text_into_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = splitter.split_text(text)
    return text_chunks


def build_vectorstore(text_chunks):
    embedder = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedder)
    return vectorstore


def create_conversational_chain(vectorstore):
    llm_model = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory_store = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=vectorstore.as_retriever(),
        memory=memory_store
    )
    return conversation_chain


def process_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for index, message in enumerate(st.session_state.chat_history):
        if index % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        process_user_input(user_question)

    with st.sidebar:
        st.subheader("Upload Your Documents")
        pdf_files = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Extract text from PDFs
                raw_text = extract_pdf_text(pdf_files)

                # Split text into chunks
                text_chunks = split_text_into_chunks(raw_text)

                # Build vector store
                vectorstore = build_vectorstore(text_chunks)

                # Initialize conversation chain
                st.session_state.conversation = create_conversational_chain(vectorstore)


if __name__ == '__main__':
    main()
