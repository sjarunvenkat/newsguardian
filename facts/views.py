from django.shortcuts import render
import pandas as pd
import numpy as np
import googlesearch_py
from sentence_transformers import SentenceTransformer, util
from .forms import factsForm
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import io
import urllib, base64
model = SentenceTransformer('stsb-roberta-large')
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import inspect
from googlesearch import search
import requests
import datetime
import time
import spacy
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import nltk
nltk.download('punkt')
classifier = pipeline('zero-shot-classification')

def home(request):
    if request.method == 'POST':
        form = factsForm(request.POST)
        if form.is_valid():
            facts = form.cleaned_data['facts']
        query = facts
        num_results = 5
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

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        texts = []
        for i in range(len(top_urls)):
            response = requests.get(top_urls[i], headers=headers)

            soup = BeautifulSoup(response.content, 'html.parser')

            text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
            # texts.append(text)
            texts.append(text)

        LANGUAGE = "english"
        SENTENCES_COUNT = 100
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
            print("Similarity score:", similarity)
            similarities.append(similarity)
           

        sentences = np.array(summaries)
        text = ' '.join(sentences)
        wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)

        fig = plt.figure(facecolor=None)
        fig = plt.imshow(wordcloud)
        fig = plt.axis("off")
        fig = plt.tight_layout(pad=0)

        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf,format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        eight = []

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

            similarity = cosine_similarity(embeddings)[0][1]

            # Print the similarity score
            #   print("Similarity score:", similarity)
            similarities.append(similarity)

        # for i in range(0,len(desc)):
        #     sen1 = model.encode(facts, convert_to_tensor=True)
        #     sen2 = model.encode(desc[i], convert_to_tensor=True)
        #     cosine_scores = list(util.pytorch_cos_sim(sen1, sen2).reshape(-1).tolist())
        #     eight.append(cosine_scores)
        
        nine = []

        # for i in range(0, len(eight)):
        #     nine.append(eight[i][0])

        # avg = sum(nine) / len(nine)

        avg = np.mean(similarities)

        threshold = 0.38
        
        addme = []

        if avg < threshold :
            for i in range(0,len(top_urls)):
                fine = top_urls[i]
                addme.append(fine)
            addme = pd.DataFrame(addme, columns=[' ']).set_index([' '])
            print(addme.columns)
            addme = addme.to_html()
            fact_check = "We can classify the news as Fake"

        else:
            for i in range(0,len(top_urls)):
                fine = top_urls[i]
                addme.append(fine)
            addme = pd.DataFrame(addme, columns=[' ']).set_index([' '])
            print(addme.columns)
            addme = addme.to_html()
            # Define the candidate labels for classification
            labels = ['hate speech', 'non-hate speech']

            # Define the text to be classified
            text = query

            # Perform zero-shot classification
            result = classifier(text, labels)

            # Get the top candidate label and its score
            top_label = result['labels'][0]
            top_score = result['scores'][0]

            # Print the result
            if top_label == 'hate speech':
                fact_check = "We can classify the news as Fake"
            else:
                labels = ['Not Profane', 'Profane']

                # Define the text to be classified
                text = query

                # Perform zero-shot classification
                result = classifier(text, labels)

                # Get the top candidate label and its score
                top_label = result['labels'][0]
                top_score = result['scores'][0]

                if top_label == 'Profane':
                   fact_check = "We can classify the news as Fake"
                else:
                    fact_check = "The News is True"
                        

        return render(request,'index.html',{'facts':form,'fine':addme,'fact_check':fact_check,'data':uri})
    
    else:
        form = factsForm()
        return render(request,'index.html',{'facts':form})
        
    