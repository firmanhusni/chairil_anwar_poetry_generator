def poetry_gen(input_text, num_words):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    import numpy as np 
    
    file = "C:/Users/Firman/Python_Test/chairil_anwar.txt"
    tokenizer = Tokenizer()
    data = open(file).read()
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    for line in corpus:
    	token_list = tokenizer.texts_to_sequences([line])[0]
    	for i in range(1, len(token_list)):
    		n_gram_sequence = token_list[:i+1]
    		input_sequences.append(n_gram_sequence)
    
    # pad sequences 
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    new_model = tf.keras.models.load_model("C:/Users/Firman/Python_Test/saved_model.ckpt")
    
      
    for _ in range(num_words):
    	token_list = tokenizer.texts_to_sequences([input_text])[0]
    	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    	predicted = new_model.predict_classes(token_list, verbose=0)
    	output_word = ""
    	for word, index in tokenizer.word_index.items():
    		if index == predicted:
    			output_word = word
    			break
    	input_text += " " + output_word
    return input_text

print('Enter the first words')
x = input()
print('\nHow many words to be generated?')
y = int(input())

result = poetry_gen(x,y)
print('\nHere is the poetry for you : \n', result)