import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec


import os
import heapq



def make_list_of_sentence(sentnce):
    tokens = sentnce.strip().split(' ')
    hebrew_letters = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י',
    'כ', 'ך', 'ל', 'מ', 'ם', 'נ', 'ן', 'ס', 'ע', 'פ',
    'ף', 'צ', 'ץ', 'ק', 'ר', 'ש', 'ת']
    return_list = []
    for token in tokens: 
        if token =='':
            continue
        #this means that the current tokent is a short cut and we should include them if the fist charechter is a hebrew latter
        #and the second the second to last latter must be also hebrew latter
        if token[-1] == "'":
            if len(token)>1 and token[-2] in hebrew_letters and token[0] in hebrew_letters:
                return_list.append(token)
        elif token[-1] in hebrew_letters and token[0] in hebrew_letters:
            return_list.append(token)
    return return_list


#the main function for section 1 ,part 1
def make_list(sentences):
    return_list = []
    for sentence in sentences:
        return_list.append(make_list_of_sentence(sentence))
    return return_list


def Section2_part1(dir,model):
    word_list = ['ישראל', 'כנסת', 'ממשלה', 'חבר', 'שלום', 'שולחן']
    text=''
    for word1 in word_list:
        text+=word1+': '
        for word2 in word_list:
            text+=f'({word2},{model.wv.similarity(word1,word2)}),'
        text = text[:-1]+'\n'
    text = text[:-1]
    with open(os.path.join(dir,'knesset_similar_words.txt'),'w',encoding='utf-8') as file:
        file.write(text)


def embeddings_of_sentences(sententces,model):
    sentence_dir = []
    for i,sentence in enumerate(sententces):
        tokens = make_list_of_sentence(sentence)
        avarage = np.zeros(100)
        for token in tokens:
            avarage += model.wv[token]
        avarage = avarage / len(token)
        sentence_dir.append(avarage)
    sentence_dir = np.array(sentence_dir)
    return sentence_dir
            

def section2_part4(dir,model):
    sentences = [
    "ברוכים הבאים , הכנסו בבקשה [לחדר] .",
    "אני [מוכנה] להאריך את [ההסכם] באותם תנאים .",
    "בוקר [טוב] , אני [פותח] את הישיבה .",
    "שלום] , הערב התבשרנו שחברינו [היקר] לא ימשיך איתנו [בשנה] הבאה] ."
    ]

    room = model.wv.most_similar(positive=['אולם',"שתכנס"],negative=[], topn=1000)#the thierd one gives "to the home"
    ready = model.wv.most_similar(positive=['יכול','מוכנה'],negative=[], topn=3)#the thired one gives יכולה
    good = model.wv.most_similar(positive=['מציון'],negative=[], topn=100)#the thired one gives ready but for male
    
    print(room)




if __name__ == '__main__':

    dir = ''
    data = pd.read_csv('knesset_corpus.csv')
    #section 1 part 1
    '''
    tokenized_sentences  = make_list(data['sentence_text'])

    #section 1 part 2
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)
    #model.train(['ישראל'],epochs=3)
    model.save("knesset_word2vec.model")'''
    model = Word2Vec.load("knesset_word2vec.model")

    

    #section 2
    
    #section 2, part 1
    Section2_part1(dir,model)

    #section 2, part 2 and 3
    sentence_embeddings = embeddings_of_sentences(data['sentence_text'],model)
    #sentences_index = [75158,679,75880,76626,77589,1330,368,77840,78170,78304]
    hebrew_sentences = [
    "אבל זה אותו דבר .",
    "ולכן , צריך להביא את זה בחשבון .",
    "אני לא מוכן לקבל את זה .",
    "אם כן , רבותי, אנחנו עוברים להצבעה .",
    "זה לא דבר שהוא חדש .",
    "מה התפקיד שלכם בנושא הזה ?",
    "בגלל שאני אומר את האמת ?",
    "בכל מקרה ההצבעה לא תתקיים היום .",
    "אני לא כל כך מבין .",
    "איך ייתכן דבר כזה ?"
    ]

    text = ""
    our_index_embeddings = embeddings_of_sentences(hebrew_sentences,model)
    matrix = cosine_similarity(our_index_embeddings,sentence_embeddings)
    '''
    for i,index in enumerate(sentences_index):
        text+=data.iloc[index]['sentence_text']+': most similar sentence: '
        max_index = matrix[i].argsort()[-2]
        text+=data.iloc[max_index]['sentence_text']+'\n'
    text = text[:-1]
    '''
    
    for i,index in enumerate(our_index_embeddings):
        text+=hebrew_sentences[i]+': most similar sentence: '
        max_index = matrix[i].argsort()[-2]
        text+=data.iloc[max_index]['sentence_text']+'\n'
    text = text[:-1]
    with open(os.path.join(dir,'knesset_similar_sentences.txt'),'w',encoding='utf-8') as file:
        file.write(text)
    
    #section2, part4
    section2_part4(dir,model)
    
    
    

