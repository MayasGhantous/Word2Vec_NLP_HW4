from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
file_path = 'masked_sentences.txt'
model.eval()

with open(file_path,'r',encoding='utf-8') as file:
    text = file.read()
    new_text = ''
    sentences = text.split('\n')
    for sentence in sentences:
        new_text +="Original sentence: "+sentence+'\n'
        sentence_Mask = sentence.replace('[*]','[MASK]')
        output = model(tokenizer.encode(sentence_Mask, return_tensors='pt'))
        print(output)
        