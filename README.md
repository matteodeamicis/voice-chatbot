# voice chatbot
To properly execute the solution, you need to follow the following instructions:

1. Make sure you have Python installed on your system. You can download it from the official Python website (https://www.python.org/) and follow the appropriate installation instructions for your operating system.

2. Create a new virtual environment (optional, but recommended) to isolate the project dependencies. You can use tools like "virtualenv" or "conda" to create a virtual environment.

3. Activate the virtual environment to work within it.

4. Install the necessary libraries by running the command:

   pip install streamlit PyPDF2 transformers langchain ipywidgets gtts pydub pydub play speechrecognition pyttsx3

5. Make sure you have a valid OpenAI API key. You can obtain it from the OpenAI website (https://platform.openai.com/docs/guides/authentication) by following the appropriate instructions.

6. Copy and paste the provided code into a Python file with the ".py" extension.

7. Save the Python file.

8. Open the terminal or command line and navigate to the directory where the Python file is located.

9. Execute the following command to start the Streamlit application, making sure to replace "filename.py" with the actual name of the Python file:

   streamlit run filename.py

10. A local server will be started, and the Streamlit application will be accessible at "http://localhost:8501" in your default browser.

11. Follow the instructions displayed in the user interface to enter your OpenAI API key in the box located in the sidebar and press Enter to confirm the input. At this point, upload a PDF file (up to a maximum size of 200 MB), and once the upload is complete, HAL will start automatically. It will respond to all your questions until you say the command 'stop' to end the conversation.
