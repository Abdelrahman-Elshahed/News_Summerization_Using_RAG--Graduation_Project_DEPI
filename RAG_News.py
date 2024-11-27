import requests
from bs4 import BeautifulSoup
import os
import replicate
from transformers import pipeline
import chromadb
from chromadb.utils import embedding_functions
import xml.etree.ElementTree as ET
import pandas as pd
import envs

class WebScraper:
    def __init__(self, url, headers=None):
        self.url = url

    @staticmethod
    def extract_paragraphs(html_content):
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            paragraph = [p.text for p in soup.find_all('p')]
            return paragraph
        else:
            return []

    def fetch_page(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch the page. Status code: {response.status_code}")
            return None

    def fetch_and_extract_p(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            paragraph = self.extract_paragraphs(response.text)
            return " ".join(paragraph)
        else:
            print(f"Failed to fetch the page. Status code: {response.status_code}")
            return None

class NYTimesAPI:
    def __init__(self):
        self.api_key = envs.NYT_API  
        self.base_url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'  # default base url

    def get_response(self, news_topic) -> list:
        url = f'{self.base_url}?q={news_topic}&api-key={self.api_key}'
        response = requests.get(url).json()
        if 'response' in response and 'docs' in response['response']:
            docs = response['response']['docs']
            print(docs)
            abstract = docs[0].get('abstract', '')
            snippet = docs[0].get('snippet', '')
            lead_paragraph = docs[0].get('lead_paragraph', '')
            result = abstract + ' ' + snippet + ' ' + lead_paragraph
            return result
        return []

class TextSummarizationPipeline:
    def __init__(self, model_name="dhivyeshrk/bart-large-cnn-samsum"):
        self.pipe = pipeline("text2text-generation", model=model_name)

    def generate_summary(self, input_text):
        if isinstance(input_text, list):
            input_text = ' '.join(input_text)  
        words = input_text.split(" ") 
        if len(words) > 500:
            input_text = " ".join(words[:500])
        return self.pipe(input_text)

class ReplicateAPI:
    """
    High Level and Scalable API for accessing any model hosted on Replicate AI.
    """
    def __init__(self, model_name, api_token=envs.REPLICATE_API):
        os.environ['REPLICATE_API_TOKEN'] = api_token
        self.model_name = model_name
        self.input_params = {
            "top_k": 0,  
            "top_p": 0.95,  
            "prompt": "",
            "max_new_tokens": 10,  
            "temperature": 0.2,  
            "length_penalty": 1,  
            "presence_penalty": 0,  
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",  
            "prompt_template": '''
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a helpful assistant. 
                <|eot_id|><|start_header_id|>user<|end_header_id>

                {prompt}
                \nPlease respond in the format: [number, '.', 'category_name'].
                \nWhich of the following classes does the above statement fall into: 
                1. Technology
                2. Sports
                3. Science
                4. Health
                5. General
                \nEnsure your response contains both the number and the full category name, like this: [3, '.', 'Science'].
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            '''
        }

    def run_model(self, prompt) -> str:
        """
        Run the model with the provided prompt and return the categorized output.
        """
        self.input_params['prompt'] = prompt

        try:
            out = replicate.run(self.model_name, input=self.input_params)
            

            if isinstance(out, list) and len(out) > 0:
                result = ''.join(out).strip() 
                return result
            else:
                print("Unexpected output format:", out)
                return ""
        except Exception as e:
            print(f"Error occurred: {e}")
            return ""

class XMLParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.root = None
        self.data = []

    def parse_xml(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        self.root = ET.fromstring(xml_content)

    def extract_information(self):
        if self.root is None:
            raise ValueError("XML not parsed. Call parse_xml() first.")
        for item in self.root.findall('.//item'):
            title = item.find('title').text
            link = item.find('link').text
            description = item.find('description').text

            domains = [category.text for category in item.findall('.//category[@domain]')]

            item_info = {
                'title': title,
                'link': link,
                'description': description,
                'domains': domains
            }
            self.data.append(item_info)

        return self.data

def add_embeddings(collection_name, xml_filepath):
    xml_parser = XMLParser(xml_filepath)
    xml_parser.parse_xml()
    result = xml_parser.extract_information()

    for ind, res in enumerate(result):
        domains = ", ".join(res['domains'])
        collection_name.add(
            documents=f"{res['title']} Domains : {domains}",
            metadatas=[{'link': res['link']}],
            ids=[f'id{ind}']
        )

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="DataBase_ChromaDB/data")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/sentence-t5-base")

    health_col = client.get_or_create_collection(name="Health", embedding_function=sentence_transformer_ef)
    science_col = client.get_or_create_collection(name="Science", embedding_function=sentence_transformer_ef)
    sports_col = client.get_or_create_collection(name="Sports", embedding_function=sentence_transformer_ef)
    tech_col = client.get_or_create_collection(name="Technology", embedding_function=sentence_transformer_ef)

    add_embeddings(health_col, 'DataBase_ChromaDB\News _Categorization_Files_XML/Health.xml')
    add_embeddings(science_col, 'DataBase_ChromaDB\News _Categorization_Files_XML/Science.xml')
    add_embeddings(sports_col, 'DataBase_ChromaDB\News _Categorization_Files_XML/Sports.xml')
    add_embeddings(tech_col, 'DataBase_ChromaDB\News _Categorization_Files_XML/Technology.xml')

    print(health_col.peek())

def get_linksDB(collection_name, prompt) -> list:
    client = chromadb.PersistentClient(path="DataBase_ChromaDB/data")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/sentence-t5-base")
    collection_name = collection_name.capitalize()
    
    try:
        db_collection = client.get_collection(name=f"{collection_name}", embedding_function=sentence_transformer_ef)
    except InvalidCollectionException:
        print(f"Collection {collection_name} does not exist.")
        return []

    result = db_collection.query(
        query_texts=[prompt],
        n_results=3
    )
    related_links = [i['link'] for i in result['metadatas'][0]]

    return related_links

def categorize(prompt: str, model: str) -> str:
    api = ReplicateAPI(model_name=model)
    output = api.run_model(prompt)

    print("Model output:", output)

    expected_format = None
    categories = ['Technology', 'Science', 'Health', 'Sports']

    for i, category in enumerate(categories, start=1):
        if category.lower() in output.lower():  # Case insensitive match
            expected_format = f"{category}"
            break

    if expected_format:
        return expected_format
    else:
        print("No matching category found. Returning empty string.")
        return ""

def get_news(url: str) -> list:
    if 'www.nytimes.com' in url:
        scraper = NYTimesAPI()
        news = scraper.get_response(url)
        return news
    else:
        scraper = WebScraper(url)
        return scraper.fetch_and_extract_p()
def get_news_GEN(url: str, links: str) -> list:
    if 'www.nytimes.com' in url:
        scraper = NYTimesAPI()
        news = scraper.get_response(links)
        return news
    else:
        scraper = WebScraper(url)
        return scraper.fetch_and_extract_p()
if __name__ == "__main__":
    news_topic = "latest in health"
    print("News from NY Times:")
    print(get_news('https://www.nytimes.com/'))
    print("Categorizing the news:")
    print(categorize(news_topic, model='meta/meta-llama-3-8b-instruct'))
