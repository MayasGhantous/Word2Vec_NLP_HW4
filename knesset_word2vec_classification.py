import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.model_selection import cross_val_predict,train_test_split
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
import sys
chunk_size = 1
def process(group):
    try:
        
        
        sentences = group['sentence_text'].tolist()
        #create the data
        row={}
        row['protocol_type']= [group.iloc[0]['protocol_type'] for _ in range(0, len(sentences)-(chunk_size-1), chunk_size)]

        #compine the each 5 sentences
        combined = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences)-(chunk_size-1), chunk_size)]
        row['sentence_text'] = combined


        #convert to a data frame
        data_frame = pd.DataFrame(row)
        return data_frame
    except Exception as ex:
        print(f'Exception at process: {ex}')

def make_chunks(data):
    try:
        #devide the data into the right groups (according to protocol_name and the type) and apply procces for each group
        #the func in DataFrameGroupBy.apply(func), func takes a data frame and can return a data frame and 
        #thats why this code works because process return a data frame
        result_df = data.groupby(['protocol_type','protocol_name'],dropna = True).apply(process).reset_index(drop = True)

        return result_df
    except Exception as ex:
        print(f'Exception at make_chunks: {ex}')




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
            elif token[-1] in hebrew_letters and token[0] in hebrew_letters:
                return_list.append(token)
        return return_list
    except  Exception as ex:
        print(f'Exception at make_list_of_sentence: {ex}')


def embeddings_of_sentences(sententces,model):
    try:
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
    except Exception as ex:
        print(f'Exception at embeddings_of_sentences: {ex}')

def down_sample(data,N):
    try:
        # if we want to down sample non positive number then dont do any thing
        if N<=0:
            return data
        number_list = random.sample(range(len(data)),k=N)
        return data.drop(number_list).reset_index(drop = True)
    except Exception as ex:
        print(f'Exception at down_sample: {ex}')

if __name__=='__main__':
    try:
        if len(sys.argv) != 3:
            print('must have 2 args')
            exit(1)
        
        data_path = sys.argv[1]
        model_path = sys.argv[2]
        chunk_sizes = [1,3,5]
        for main_chunk_size in chunk_sizes:

            print(f'for chunk size {main_chunk_size}')
            chunk_size = main_chunk_size
            
            model = Word2Vec.load(model_path)
            df = pd.read_csv(data_path)
            #make chunks
            df = make_chunks(df)

            committee_data = df.loc[df['protocol_type'] == 'committee'].reset_index(drop=True)
            plenary_data = df.loc[df['protocol_type'] == 'plenary'].reset_index(drop=True)

            #plenary_data = process(plenary_data)
            #committee_data = process(committee_data)

            #down sample the data
            plenary_data = down_sample(plenary_data,len(plenary_data) -len(committee_data))
            committee_data = down_sample(committee_data,len(committee_data)-len(plenary_data))

            data = pd.concat([plenary_data,committee_data])
            data = data.sample(frac=1,random_state=42).reset_index(drop = True)

            labels = data['protocol_type']
            features = embeddings_of_sentences(data['sentence_text'],model)

            KNN = KNeighborsClassifier(50)
            print(f'KNN with corss validation: ')
            KNN_cross_validation = cross_val_predict(KNN,features,labels,cv=10)
            print(sklearn.metrics.classification_report(labels, KNN_cross_validation))

            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42,stratify=labels)
            KNN.fit(X_train,y_train)
            print(f'KNN with split: ')
            y_pred = KNN.predict(X_test)
            print(sklearn.metrics.classification_report(y_test, y_pred))
    except Exception as ex:
        print(f'Exception at main: {ex}')
