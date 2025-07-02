import os
from dotenv import load_dotenv
from sympy import true

# Load environment variables from .env file
load_dotenv()

# Now you can access the environment variables using os.getenv() or os.environ.get()
# For Gemini API Key (using a consistent name from .env)
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ['GOOGLE_API_KEY'] = gemini_api_key # Many Google libraries expect GOOGLE_API_KEY
    # or if a specific library needs 'geminikey' directly:
    # os.environ['geminikey'] = gemini_api_key
else:
    print("GEMINI_API_KEY not found in .env file.")



#-------------------------doc upload stream lit------------------------------------------------------------------------
# file load
import streamlit as st
files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader(files)  # Adjust path as needed
# pages = loader.load()






from langchain_community.document_loaders import PDFPlumberLoader
all_pages = []
# for file in files:
#     loader = PDFPlumberLoader(file)
#     pages = loader.load()
#     all_pages.extend(pages)

#     st.success(f"Loaded {len(all_pages)} pages from {len(files)} files.")

if files:
    for file in files:
        # Save to temp file
        temp_path = os.path.join("temp_dir", file.name)
        
        # Make sure temp_dir exists
        os.makedirs("temp_dir", exist_ok=True)
        
        # Write uploaded file to disk
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Now load using the file path
        loader = PDFPlumberLoader(temp_path)
        pages = loader.load()
        all_pages.extend(pages)
    
    st.success(f"Loaded {len(all_pages)} pages from {len(files)} files.")




# spliting
if files:

    from langchain_text_splitters import RecursiveCharacterTextSplitter



    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
    )
    page_texts = [page.page_content for page in all_pages]

# Pass the list of strings to create_documents
    texts = text_splitter.create_documents(page_texts)


# embadding

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Use the correct embedding class
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ['GOOGLE_API_KEY'])
# %%
    from langchain_community.vectorstores import Chroma

    db = Chroma.from_documents(texts, embedding=embeddings)



# getting prompt
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")


# llm model define
    from langchain_google_genai import GoogleGenerativeAI

    llm=GoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.1, max_output_tokens=20000, google_api_key=os.environ['GOOGLE_API_KEY'])


# formating pages
    def format_docs(pages):
        return "\n\n".join(
             f"[{i+1}] {page.metadata.get('source', '')}\n{page.page_content}"
        for i, page in enumerate(pages)
    )

#retrivel + rag chain
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    retriever=db.as_retriever()
    rag_chain=({"context": retriever|format_docs, "question": RunnablePassthrough()} | prompt | llm|StrOutputParser())
#  -----------------------------asking question--------------------------------------------------------
    # question="hi"
    # i=0
    # while(question!="none"):
    #      question = st.text_area("Enter your question:", key=i)
    #      st.write("Model output:")
    #      if question:
    #         st.write(rag_chain.invoke(question))
    #         print(rag_chain.invoke(question))
    #         i=i+1
    # Initialize session state to keep track of conversation
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Ask Me Anything")

# Input for user's question
question = st.text_input("Enter your question:")

# Show response only if a question is entered
if question:
    # Call your RAG chain or model here
    answer = rag_chain.invoke(question)

    # Store the Q&A pair in session state
    st.session_state.history.append((question, answer))

    # Clear the input box (optional – needs form for perfect reset)
    # st.experimental_rerun()

# Display previous Q&A
if st.session_state.history:
    st.subheader("Conversation History:")
    for i, (q, a) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
            