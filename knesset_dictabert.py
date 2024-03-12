from transformers import AutoModelForMaskedLM, AutoTokenizer


if __name__ == '__main__':
    try:
        tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
        model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
        file_path = 'masked_sentences.txt'
        output_file_path = 'dictabert_results.txt'
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
                    current_sentence = current_sentence.replace('[*]','[MASK]',1)
                    MASK_index = current_sentence.split(' ').index('[MASK]')
                    tokenizes = tokenizer.encode(current_sentence, return_tensors='pt')
                    output = model(tokenizes)



                    index = output.logits[0, MASK_index+1, :].argmax()
                    predicted_token = tokenizer.convert_ids_to_tokens([index])[0]
                    current_sentence = current_sentence.replace('[MASK]',predicted_token)
                    token_list.append(predicted_token)
                new_text += 'DictaBERT sentence: '+current_sentence+'\n'
                new_text +='DictaBERT tokens: ' +', '.join(token_list) +'\n'
            with open(output_file_path,'w',encoding='utf-8') as output_file:
                output_file.write(new_text)
    except Exception as ex:
        print(f'exception at main: {ex}')

