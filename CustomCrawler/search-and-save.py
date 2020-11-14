from requests import get 
import json
from readability import Document
import time
from rake_nltk import Rake


class SearchArticle : 
    
    def __init__(self) : 
        self.url = "https://www.googleapis.com/customsearch/v1?key=AIzaSyDscip5huSYdv7NWiGStVRSnfmMDdj0YlE&cx=016577003490921499880:wrypi5_dymm&q="
        
    def search(self, keywords) :
        
        google_link = self.url + keywords
        res = get(google_link)
        json_data = json.loads(res.text)
        
        article_links = []
        for i in range(len(json_data['items'])) : 
            article_links += [json_data['items'][i]['link']]
        
        data = []
        for i in article_links : 
            print(i)
            article = Article(i)
            article.download()
            article.parse()            
            data += [[article.title, article.text]]
            
        return data
    def get_key_words(text):
        r = Rake()
        r.extract_keywords_from_text(text)
        keys = r.get_ranked_phrases_with_scores()
        return keys[0][1]

