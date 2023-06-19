import streamlit as st
import os
import tempfile
import PyPDF2
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from IPython.display import display, Audio
import ipywidgets as widgets
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr

# Step 1: Prompt user to enter the OPENAI_API_KEY
api_key = st.sidebar.text_input("Inserisci la tua OpenaAI API Key 🔑", type="password",
                                placeholder="Inserisci qui la tua OpenAI API Key (sk-...)",
                                help="Puoi ottenere la tua API Key da https://platform.openai.com/account/api-keys.")
st.sidebar.markdown("---")
# Set the environment variable with the provided API key
os.environ["OPENAI_API_KEY"] = api_key


st.sidebar.title("Strumenti utilizzati")
st.sidebar.markdown('''
         ## About
         Quest'app è un ChatBot vocale basato su LLM, costruito utilizzando:
         - [Streamlit](https://streamlit.io/)
         - [LangChain](https://python.langchain.com/)
         - [OpenAI](https://platform.openai.com/docs/models) LLM model
         ''')
st.sidebar.markdown("---")
st.sidebar.markdown("Creato da Matteo De Amicis & Gianmarco Venturini 🤝")
st.title("ChatBot HAL 🤖")
st.subheader("Benvenuto! Sono qui per rispondere alle tue domande sul file PDF")

# Step 1: Convert PDF to text
uploaded_file = st.file_uploader("Carica qui il tuo file PDF 📄", type="pdf")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    with open(tmp_file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    
            
    # Step 3: Create function to count tokens
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    # Step 4: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=512,
        chunk_overlap=24,
        length_function=count_tokens,
    )

    chunks = text_splitter.create_documents([text])

    type(chunks[0])

    
    # Get embedding model
    embeddings = OpenAIEmbeddings()

    # Create vector database
    db = FAISS.from_documents(chunks, embeddings)

    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    query = " "
    docs = db.similarity_search(query)

    chain.run(input_documents=docs, question=query)

    # Create conversation chain that uses our vectordb as retriever, this also allows for chat history management
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

    
    chat_history = []
    r = sr.Recognizer()  # Declare r globally

    def on_submit(query):
        
        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result['answer']))

        st.markdown(f"**Utente:** {query}")
        st.markdown(f"**<span style='color:blue'>HAL</span>:** {result['answer']}", unsafe_allow_html=True)

        risposta_vocale = converti_in_vocale(result['answer'])
        riproduci_audio(risposta_vocale)

        process_audio_input()  # Riattiva il microfono per un'altra domanda

    def recognize_query(audio):
        try:
            # Trasforma l'audio in testo
            text = r.recognize_google(audio, language='it-IT')
            if 'stop' in text.lower():
                st.markdown(f"**<span style='color:blue'>HAL</span>:** Conversazione interrotta. Spero di essere stato d'aiuto!", unsafe_allow_html=True)
                return
            on_submit(text)
        except sr.UnknownValueError:
            st.markdown(f"**HAL:** Non sono riuscito a comprendere l'audio")
            process_audio_input()
        except sr.RequestError as e:
            st.text("Errore durante la richiesta a Google Speech Recognition service: {0}".format(e))
            process_audio_input()

    def process_audio_input():
        # Utilizza il microfono come sorgente audio
        with sr.Microphone() as source:
            st.markdown("**<span style='color:blue'>HAL</span>:** Parla, ti ascolto! Pronuncia **<span style='color:red'>stop</span>** per interrompere la conversazione con HAL.", unsafe_allow_html=True)
            audio = r.listen(source)
            recognize_query(audio)

    def on_button_clicked():
        process_audio_input()

    def converti_in_vocale(testo):
        tts = gTTS(text=testo, lang='it')
        tts.save("risposta_vocale.mp3")
        return "risposta_vocale.mp3"

    def riproduci_audio(file_audio):
        audio = AudioSegment.from_file(file_audio)
        play(audio)

    def main():

        process_audio_input()

    if __name__ == '__main__':
        main()


