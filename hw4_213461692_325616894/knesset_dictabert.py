from transformers import AutoModelForMaskedLM, AutoTokenizer
import os
import torch
import sys


if __name__ == '__main__':
    try:
        if  len(sys.argv) !=3:
            print('must have 2 args')
            exit(1)
        
        tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
        model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
        dir = sys.argv[2]
        file_path = sys.argv[1]
        dir=''
        file_path = 'masked_sentences.txt'
        output_file_path = os.path.join(dir,'dictabert_results.txt')
        model.eval()

        with open(file_path,'r',encoding='utf-8') as file:
            text = file.read()
            #text = 'בשנת 1948 השלים אפרים קישון את [MASK] בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
            new_text = ''
            sentences = text.split('\n')[:-1]
            for sentence in sentences:
                token_list = []

                current_sentence = sentence
                new_text +="Original sentence: "+sentence+'\n'
                sentence_Mask =''
                count = sentence.count('[*]')
                for _ in range(count):
                    #change just the first [*] because we are working with one mask at a time
                    current_sentence = current_sentence.replace('[*]','[MASK]',1)
                    MASK_index = current_sentence.split(' ').index('[MASK]')
                    tokenizes = tokenizer.encode(current_sentence, return_tensors='pt')
                    output = model(tokenizes)



                    index = torch.topk(output.logits[0, MASK_index+1, :],1)[1]
                    predicted_token = tokenizer.convert_ids_to_tokens([index])[0]
                    current_sentence = current_sentence.replace('[MASK]',predicted_token)
                    token_list.append(predicted_token)
                new_text += 'DictaBERT sentence: '+current_sentence+'\n'
                new_text +='DictaBERT tokens: ' +', '.join(token_list) +'\n'+'\n'
            with open(output_file_path,'w',encoding='utf-8') as output_file:
                new_text = new_text[:-2]
                output_file.write(new_text)
    except Exception as ex:
        print(f'exception at main: {ex}')

