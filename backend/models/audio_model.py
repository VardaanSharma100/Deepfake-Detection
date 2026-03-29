from langchain_core.prompts import ChatPromptTemplate
from data.audio.audio_data import audio_data
from preprocessing import text_preprocessing
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import assemblyai as aai
import os
from logs.audio_logging import audio_logger

logger = audio_logger().get_logger()

class audio_model():
    def __init__(self, api_key, audio_key):
        self.api_key = api_key
        self.parser = StrOutputParser()
        self.model = ChatGroq(model='llama3-70b-8192', api_key=self.api_key)
        aai.settings.api_key = audio_key
        self.transcriber = aai.Transcriber()
        self.response = None  

    def predict(self, query):
        try:
            self.query = self.transcriber.transcribe(query)
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            self.response = "Transcription failed"
            return self.response

        try:
            self.load_text = audio_data(self.query.text)
            self.data = self.load_text.load_data()
            self.folder_path = getattr(self.load_text, 'folder_path', 'output/audio')
            self.audio_file = os.path.join(self.folder_path, 'audio_query.mp3')
            self.timestamp = self.load_text.timestamp
        except Exception as e:
            logger.warning(f"Loading audio data failed: {e}")
            self.response = "Audio data load failed"
            return self.response

        try:
            os.makedirs(self.folder_path, exist_ok=True)
            with open(self.audio_file, 'wb') as f:
                f.write(query.read())
        except Exception as e:
            logger.warning(f"Cannot open/write file: {e}")

        try:
            self.preprocessing_model = text_preprocessing(self.load_text)
            self.cleaned = self.preprocessing_model.preprocess_data(self.data)
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            self.cleaned = ""

        try:
            self.systemprompt = (
                "You are a fast news checking AI. Analyze the provided web-scraped data "
                "and answer only in one word: Real or Fake."
            )
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ('system', self.systemprompt),
                    ('system', '{data}\n'),
                    ('human', '{query}')
                ]
            )
            self.chain = self.prompt | self.model | self.parser
            self.response = self.chain.invoke({'data': self.cleaned, 'query': self.query.text})
        except Exception as e:
            logger.warning(f"Model invocation failed: {e}")
            self.response = "Model invocation failed"

        try:
            os.makedirs('output/audio', exist_ok=True)
            self.result = os.path.join('output/audio', f'run_{self.timestamp}_result.txt')
            with open(self.result, 'w', encoding='utf-8') as f:
                f.write(self.response)
        except Exception as e:
            logger.warning(f"Saving result failed: {e}")

        return self.response
