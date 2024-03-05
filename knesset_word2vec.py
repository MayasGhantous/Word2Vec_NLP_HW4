import pandas as pd
import numpy as np
import os


from gensim.models import Word2Vec

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
    sentence_dir = {}
    for i,sentence in enumerate(sententces):
        tokens = make_list_of_sentence(sentence)
        avarage = np.zeros(100)
        for token in tokens:
            avarage += model.wv[token]
        avarage = avarage / len(token)
        sentence_dir[i] = avarage
    return sentence_dir
            





if __name__ == '__main__':

    #section 1 part 1
    data = pd.read_csv('knesset_corpus.csv')
    tokenized_sentences  = make_list(data['sentence_text'])

    #section 1 part 2
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)
    #model.train(['ישראל'],epochs=3)
    model.save("knesset_word2vec.model")
    

    #section 2
    #section 2, part 1
    Section2_part1('',model)
    #section 2, part 2
