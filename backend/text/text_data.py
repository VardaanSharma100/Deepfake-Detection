from googlesearch import search
from datetime import datetime
import newspaper
import os
from newspaper import Article
class text_data():
    def __init__(self,query):
        self.query=query
        self.config=newspaper.Config()
        self.config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36'
        self.base_dir='data/text'
        self.timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        self.folder_path=os.path.join(self.base_dir,f'run_{self.timestamp}')
        try:
            os.makedirs(self.folder_path,exist_ok=True)
        except Exception as e:
             pass
        self.query_file=os.path.join(self.folder_path,'query.txt')
        try:
            with open(self.query_file,'w',encoding='utf-8') as f:
                    f.write(self.query)
        except Exception as e:
            pass    
    def get_google_links(self):
        self.links=[]
        for url in search(self.query,2):
            self.links.append(url)
        self.links_file=os.path.join(self.folder_path,'links.txt')
        try:
            with open(self.links_file,'w',encoding='utf-8') as f:
                for text in self.links:
                    f.write(text+'\n')
        except Exception as e:
            pass            
        return self.links    
    def extract_article_content(self,urls):
        self.content=[]
        for url in urls: 
                    try:           
                        article=Article(url,config=self.config)
                        article.download()
                        article.parse()
                        self.content.append(article.text) 
                    except Exception as e: 
                        self.content.append("Could not fetch content")
        self.extracted_content_file=os.path.join(self.folder_path,'extracted_content.txt')
        try:
            with open(self.extracted_content_file,'w',encoding='utf-8') as f:
                for text in self.content:
                    f.write(text+'\n')
        except Exception as e:
            pass            
        return self.content   
    def get_folder_path(self):
        return self.folder_path
    def load_data(self):
        links=self.get_google_links()
        extracted_articles=self.extract_article_content(links)
        return extracted_articles
       

