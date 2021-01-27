import socket
import select
import sys
import threading
from _thread import *

HEADER = 64
FORMAT='utf-8'
DISCONNECT_MESSAGE = '!DISCONNECT'
PORT=1235
SERVER=socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

def func_topic_identification(question):
    #print('You have topic identification')
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text
    import re

    # dataset coronavirus WHO
    pd.set_option('max_colwidth', 100)  # Increase column width
    data = pd.read_excel("WHO_FAQ.xlsx")
    #data.head()
    
    def preprocess_sentences(input_sentences):
        return [re.sub(r'(covid-19|covid)', 'coronavirus', input_sentence, flags=re.I) for input_sentence in input_sentences]

    # Load module containing USE
    module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')
    response_encodings = module.signatures['response_encoder'](input=tf.constant(preprocess_sentences(data.Answer)), context=tf.constant(preprocess_sentences(data.Context)))['outputs']


    test_questions = [question]
    question_encodings = module.signatures['question_encoder'](tf.constant(preprocess_sentences(test_questions)))['outputs']
    
    # Get the responses
    test_responses = data.Answer[np.argmax(np.inner(question_encodings, response_encodings))]
        
    return test_responses
    

def func_text_similarity(question):
    
    #importing libraries
    import os
    import random
    import string
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from nltk.tokenize import word_tokenize
    
    # dataset coronavirus WHO
    pd.set_option('max_colwidth', 100)  # Increase column width
    data = pd.read_excel("Sentiment_recommendations.xlsx")
    sample = "I will answer your query"
    sample_1 = data['Context'][0]
    new_row = {'Context':question,'Answer':''}
    data = data.append(new_row, ignore_index=True)
    
    #Punctuation removal
    remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)
    
    #Tokenization
    def lemNormalize(text):
        return nltk.word_tokenize(text.lower().translate(remove_punct_dict))
    
    #Applying tfidf vectorizer
    TfidfVec = TfidfVectorizer(tokenizer =lemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(data['Context'])
    
    #Matching the context of user with dataset
    vals = cosine_similarity(tfidf[-1],tfidf)
    boolArr = (vals >0.3)
    result = np.where(boolArr)
    answer =''
    sample_2 = data['Context'][2]
    if (np.any(boolArr == True) and len(result[1]) >1):
        sample_3 = ' I am in IF condition'
        for i in range(0,(len(result[1]))-1):
            answer= answer+ data['Answer'][result[1][i]]
    else:
        answer= 'Sorry, dont understand.'
    return(answer)

def func_sentiment_analysis(question):

    import nltk
    #nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    vader = SentimentIntensityAnalyzer()
    vs = vader.polarity_scores(question)
    
    if vs['compound']>=0.05:
        answer = "Positive"
    elif (vs['compound']> -0.05 and vs['compound']< 0.05):
        answer = "Neutral"
    elif vs['compound'] <= -0.05:
        answer = "Negative"

    return answer


def handle_client(conn, addr):
    print(f'[NEW CONNECTION] {addr} connected.')
    
    connected = True
    count = 1
    option_count = 1
    covid_query_count = 1
    sentiment_count = 1
    while connected:
        
        #To handle every request of any lenght from client, HEADER is used
        msg_len = conn.recv(HEADER).decode(FORMAT)
        if(msg_len):
            msg_len = int(msg_len)
            msg = conn.recv(msg_len).decode(FORMAT)
            
            #if (msg == DISCONNECT_MESSAGE):
                #connected = False
            print(f'[{addr}] {msg}')
            
            if count == 1:
            
                code_output = "I am your chatbot. How may I help you?"
                user_answer= code_output
                conn.send(user_answer.encode(FORMAT))            
                
            else:
                
                if option_count == 1:
                    #Asking main menu questions to user
                    options = "Choose any one from below: \n a. Have any queries about Covid-19? \n OR \n b. Want suggestions in quarantine to enlighten your day? \n Type a OR b"
                    conn.send(options.encode(FORMAT))
                    option_count = option_count + 1
                    
                else:
                    
                    if msg.lower() == 'a':
                        #Asking questions related to Covid-19
                        if covid_query_count == 1:                        
                            covid_question = "You can now ask questions about Covid-19"
                            conn.send(covid_question.encode(FORMAT))
                            covid_query_count = covid_query_count + 1
                    
                    elif covid_query_count==2:
                        #Answer questions related to Covid-19
                        #covid_answer = func_text_similarity(msg)
                        covid_answer = func_topic_identification(msg)
                        conn.send(covid_answer.encode(FORMAT))

                        option_count = 1
                        covid_query_count = 1
                        sentiment_count = 1
                    
                    elif msg.lower() == 'b':
                        #Asking questions to know sentiments 
                        if sentiment_count == 1:                        
                            sentiment_question = "So, how are you feeling today?"
                            conn.send(sentiment_question.encode(FORMAT))
                            sentiment_count = sentiment_count + 1
                    
                    elif sentiment_count==2:
                        #Asking questions to tell sentiments to user
                        sentiment_answer = func_sentiment_analysis(msg)
                        if sentiment_answer =='Positive':
                            happy_count = 1
                            sentiment_output = f"I am glad to hear that. I can provide you some suggestions that may interest you. Type Y/N to continue"
                        elif sentiment_answer == 'Neutral':
                            happy_count = 0
                            sentiment_output = f"Well, then I have some suggestions that might interest you. Type Y/N to continue"
                        else:
                            happy_count = -1
                            sentiment_output = f"Well, I could assist you with some activities that may be of any interest. Type Y/N to continue"
                        
                        conn.send(sentiment_output.encode(FORMAT))
                        sentiment_count = sentiment_count + 1
                        
                        
                    elif sentiment_count == 3:
                        #Recommending alternatives to cater user sentiments
                        if msg.lower() == 'y':
                            if happy_count == 1:
                                sentiment_question = "Type any one of these categories : \n Play a game \n Listen to music \n Learn to play a musical instrument \n Learn to dance \n Latest videos of sports \n Latest news \n Watch movies"
                            
                            elif happy_count == 0:
                                sentiment_question = "Type any one of these categories : \n Play a game \n Learn to play a musical instrument \n Watch movies \n Workout videos \n Listen to music \n Latest videos of sports \n Learn to cook your favorite cuisine"
                            
                            else:
                                sentiment_question = "Select any one from these categories : \n Listen to Music \n Meditate to relax \n Talk to a Therapist \n Workout videos \n Try some DIY \n Learn a musical instrument \n Learn to dance"
                                                  
                            conn.send(sentiment_question.encode(FORMAT))
                            sentiment_count = sentiment_count + 1
                        
                        elif msg.lower() == 'n':
                            covid_question = "Redirecting to main menu. Can I?"
                            conn.send(covid_question.encode(FORMAT))
                            option_count = 1
                            sentiment_count = 1
                            
                        else:
                            covid_question = "Select the right option"
                            conn.send(covid_question.encode(FORMAT))
                            sentiment_count = 2
                    
                    elif sentiment_count == 4:
                        #Providing alternatives to cater user sentiments
                        '''
                        if happy_count == 1:
                            topic_identify = f"i want to {msg} and feeling happy."
                        else:
                            topic_identify = f"i want to {msg} and feeling sad"
                        
                        recommendation_answer = func_topic_identification(topic_identify,'b')
                        '''
                        recommendation_answer = func_text_similarity(msg)
                        conn.send(recommendation_answer.encode(FORMAT))
                        
                        option_count = 1
                        sentiment_count = 1
                        covid_query_count = 1
                    
                    
                    else:
                        #Asking to provide right input from user
                        covid_question = "Select the right option"
                        conn.send(covid_question.encode(FORMAT))
            
            
            count = count + 1
            
    conn.close()        
    
def start():
    #To start the server to handle various client request
    server.listen()
    print(f'[LISTENING] Server is listening on {SERVER}')
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn,addr))
        thread.start()
        print(f'[ACTIVE CONNECTIONS] {threading.activeCount() - 1}')
        
print('[STARTING] Server is starting....')

start()
