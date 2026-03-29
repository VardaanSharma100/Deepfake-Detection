from langchain_core.prompts import ChatPromptTemplate
from data.text.text_data import text_data
from preprocessing import text_preprocessing
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from logs.text_logging import text_logger
logger=text_logger().get_logger()
class text_model():
    def __init__(self,api_key):
        self.api_key=api_key
        self.parser=StrOutputParser()
        self.model=ChatGroq(model='llama-3.1-8b-instant',api_key=self.api_key)
    def predict(self,query):
        self.query=query
        try:
            self.load_text=text_data(self.query)
            self.data=self.load_text.load_data()
            self.timestamp=self.load_text.timestamp
        except Exception as e:
            logger.warning(str(e))
        try:    
            self.preprocessing_model=text_preprocessing(self.load_text)
            self.cleaned=self.preprocessing_model.preprocess_data(self.data)
        except Exception as e:
            logger.warning(str(e))
        try:
            self.systemprompt="You are a fast news checking ai here is provided the data from websited scraped and after that at last there will be a question from user asking for a news analyze the content and answer only in one word either Real or Fake\n Again answer in one word Real or Fake\n"
            self.prompt=ChatPromptTemplate.from_messages(
            [
                ('system',self.systemprompt),
                ('system','{data}\n'),
                ('human','{query}')
            ]
            )
            self.chain=self.prompt|self.model|self.parser
            self.response=self.chain.invoke({'data':self.cleaned,'query':self.query})
            self.result=os.path.join('output\text',f'run_{self.timestamp}_result.txt')
        except Exception as e:
            logger.warning(str(e))
        try:
            with open(self.result,'w',encoding='utf-8') as f:
                 f.write(self.response)
        except Exception as e:
             logger.warning(str(e))      
            
        return self.response