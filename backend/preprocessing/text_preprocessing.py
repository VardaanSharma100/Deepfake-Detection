import re
import os
from logs.text_logging import text_logger
logger=text_logger().get_logger()
class text_preprocessing():
    def __init__(self,model):
        try:
            self.model=model
            self.folder_path=self.model.get_folder_path()
        except Exception as e:
            logger.warning(str(e))
    def preprocess_text_data(self,extracted_articles):
        self.extracted_articles=extracted_articles
        self.cleaned=[]
        self.file_dir=os.path.join(self.folder_path,'processed_content.txt')
        for text in self.extracted_articles:
            try:
                text=text.lower()
                text=text.encode('ascii',errors='ignore').decode()
                text=re.sub(r'\s+',' ',text).strip()
                if text:
                    self.cleaned.append(text)
            except Exception as e:   
                logger.warning(str(e)) 
        for data in self.cleaned:
            if data== 'could not fetch content':
                self.cleaned.remove(data)   
        self.data='\n\n'.join(self.cleaned)
        self.data=self.data.split()
        self.data=' '.join(self.data[:3400])
        try:
            with open(self.file_dir,'w',encoding='utf-8') as f:
                f.write(self.data) 
        except Exception as e:
            logger.warning(str(e))           
        return self.data    
    def preprocess_data(self,extracted_articles):
        cleaned=self.preprocess_text_data(extracted_articles)
        return cleaned 
