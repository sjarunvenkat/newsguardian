from googlesearch import search
import requests
import datetime
import time
import spacy
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

query = "Earthquake in kashmir"
num_results = 5

# Get today's date and subtract one year from it
today = datetime.date.today()
one_year_ago = today - datetime.timedelta(days=365)

# Format the dates in the YYYYMMDD format
start_date = one_year_ago.strftime("%Y%m%d")
end_date = today.strftime("%Y%m%d")

# Construct the query string with the date range
query += f" daterange:{start_date}-{end_date}"

# Perform the search
top_urls=[]
for j in search(query, num_results=num_results):
    # perform your desired action with the search result here
    top_urls.append(j)
    
    # add a delay of 1 second between requests
    time.sleep(1)
# top_urls = list(search(query, num_results=num_results))

# print(top_urls)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
texts = []
for i in range(len(top_urls)):
  response = requests.get(top_urls[i], headers=headers)

  soup = BeautifulSoup(response.content, 'html.parser')

  text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
  # texts.append(text)
  texts.append(text)

# print(texts)

LANGUAGE = "english"
SENTENCES_COUNT = 5
summaries = []
for i in range(len(texts)):
  parser = HtmlParser.from_string(texts[i], None, Tokenizer(LANGUAGE))
  summarizer = LsaSummarizer()
  summarizer.stop_words = [' ']
  summary = summarizer(parser.document, SENTENCES_COUNT)
  # print(f"Summary for website {1}:")
  result = []
  for i in range(len(summary)):
    # print(str(summary[i]))
    result.append(str(summary[i]))
    str1 = " "
    str1 = str1.join(result)
  summaries.append(str1)

# print(summaries)

similarities = []
# Load the pre-trained model
nlp = spacy.load("en_core_web_md")
for i in range(len(summaries)):
  # Define two example sentences
  sentence1 = query
  sentence2 = summaries[i]

  # Get embeddings for the sentences
  embeddings = [nlp(sentence).vector for sentence in [sentence1, sentence2]]

  # Calculate cosine similarity between the sentence embeddings
  from sklearn.metrics.pairwise import cosine_similarity
  similarity = cosine_similarity(embeddings)[0][1]

  # Print the similarity score
#   print("Similarity score:", similarity)
  similarities.append(similarity)

print(similarities)