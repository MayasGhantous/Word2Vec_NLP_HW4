import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
import os



def make_list_of_sentence(sentnce):
    try:
            
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
            elif token[-1] in hebrew_letters and token[0] in hebrew_letters: #this is a regular word
                return_list.append(token)
        return return_list
    except Exception as ex: 
        print(f'Exception at make_list_of_sentence: {ex}')


#the main function for section 1 ,part 1
def make_list(sentences):
    try:
        return_list = []
        for sentence in sentences:
            return_list.append(make_list_of_sentence(sentence))
        return return_list
    except Exception as ex: 
        print(f'Exception at make_list: {ex}')


def Section2_part1(dir,model):
    try:
        word_list = ['ישראל', 'כנסת', 'ממשלה', 'חבר', 'שלום', 'שולחן']
        # iterate on the words and create the text and then save it
        text=''
        for word1 in word_list:
            text+=word1+': '
            for word2 in word_list:
                text+=f'({word2},{model.wv.similarity(word1,word2)}),'
            text = text[:-1]+'\n'
        text = text[:-1]
        with open(os.path.join(dir,'knesset_similar_words.txt'),'w',encoding='utf-8') as file:
            file.write(text)
    except Exception as ex: 
        print(f'Exception at Section2_part1: {ex}')
        


def embeddings_of_sentences(sententces,model):
    try:
        sentence_dir = []
        for i,sentence in enumerate(sententces):
            tokens = make_list_of_sentence(sentence)
            avarage = np.zeros(100)
            #fist we need to sum the vector of all the words 
            for token in tokens:
                avarage += model.wv[token]
            #make the avarge
            avarage = avarage / len(token)
            sentence_dir.append(avarage)
        sentence_dir = np.array(sentence_dir)
        return sentence_dir
    except Exception as ex:
        print(f'Exception at embeddings_of_sentences: {ex}')
        
            

def section2_part4(dir,model):
    try:
        sentences = [
        "ברוכים הבאים , הכנסו בבקשה [לחדר] .",
        "אני [מוכנה] להאריך את [ההסכם] באותם תנאים .",
        "בוקר [טוב] , אני [פותח] את הישיבה .",
        "שלום] , הערב התבשרנו שחברינו [היקר] לא ימשיך איתנו [בשנה] הבאה] ."
        ]
        
        sentences = [sentence.replace(']','').replace('[','') for sentence in sentences]
        room = model.wv.most_similar(positive=["שתבוא",'אולם'],negative=[], topn=3)#the first one gives לאולם
        ready = model.wv.most_similar(positive=['יכול','מוכנה'],negative=[], topn=3)#the thired one gives יכולה
        agreement = model.wv.most_similar(positive=['ההסכם'],negative=[], topn=3)#the thired one gives המהלך
        good = model.wv.most_similar(positive=["טוב","שמש"],negative=[], topn=3)#the thired one gives בריא
        word_open = model.wv.most_similar(positive=["מתחיל"],negative=[], topn=3)#the thired one gives ממשיך
        peace = model.wv.most_similar(positive=["רבותי","שלום","חברי","תודה"],negative=[], topn=3)#the thired one gives עמיתי
        dear = model.wv.most_similar(positive=["הטוב"],negative=[], topn=3)#the second give הגדול
        year = model.wv.most_similar(positive=['בשנה'],negative=[], topn=3)#thes second gives השנה


        
        room = 'לאולם'
        ready = "יכולה"
        agreement = "המהלך"
        good = 'בריא'# in attempt of getting אור
        word_open = "ממשיך"
        peace  = "עמיתי"
        dear = "הגדול"
        year = "השנה"
        changed_sentences = []
        #replace the words
        changed_sentences.append(sentences[0].replace("לחדר",room))
        changed_sentences.append(sentences[1].replace("מוכנה",ready).replace('ההסכם',agreement))
        changed_sentences.append(sentences[2].replace("טוב",good).replace('פותח',word_open))
        changed_sentences.append(sentences[3].replace("שלום",peace).replace('היקר',dear).replace('בשנה',year))

        #make the test and save it
        text = '\n'.join([f'{sentences[i]}: {changed_sentences[i]}' for i in range(len(sentences))])
        with open(os.path.join(dir,'red_words_sentences.txt'),'w',encoding='utf-8') as file:
            file.write(text)
    except Exception as ex:
        print(f'Exception at section2_part4: {ex}')

        



if __name__ == '__main__':
    try:

        if len(sys.argv) !=3:
            print('Exception must have 2 args')
            exit(1)
        
        dir = sys.argv[2]
        data_path = sys.argv[1]
        data = pd.read_csv(data_path)
        #section 1 part 1
        
        tokenized_sentences  = make_list(data['sentence_text'])

        #section 1 part 2
        #we dont save it again because it change each time
        model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)
        model.save(os.path.join(dir,'knesset_word2vec.model'))

        

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
        
        # we first worked with indexes
        #for i,index in enumerate(sentences_index):
            #text+=data.iloc[index]['sentence_text']+': most similar sentence: '
            #max_index = matrix[i].argsort()[-2]
            #text+=data.iloc[max_index]['sentence_text']+'\n'
        #text = text[:-1]
        
        
        for i,index in enumerate(our_index_embeddings):
            text+=hebrew_sentences[i]+': most similar sentence: '
            max_index = matrix[i].argsort()[-2]
            text+=data.iloc[max_index]['sentence_text']+'\n'
        text = text[:-1]
        with open(os.path.join(dir,'knesset_similar_sentences.txt'),'w',encoding='utf-8') as file:
            file.write(text)
        
        #section2, part4
        section2_part4(dir,model)


        #question part: 
        '''
        words_list =['חם',"גדול","יפה","טוב","שמח","עשיר","ראשון","אהבה"]
        op_words =['קר',"קטן","מכוער","רע","עצוב","עני","אחרון","שנאה"]
        for word_i in range(len(words_list)):
            similarty_score = model.wv.similarity(words_list[word_i],op_words[word_i])
            print(f'{words_list[word_i][::-1]},{op_words[word_i][::-1]}: {similarty_score}')
            '''
        
    except Exception as ex:
        print(f'Exception at main: {ex}')