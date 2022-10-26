from tatoebatools import ParallelCorpus, tatoeba                                #for getting tatoeba data
from unidecode import unidecode as decode                                       #decoding tatoeba data
from re import sub                                                              #formatting tatoeba data
from random import shuffle                                                      #shuffling data
from collections import Counter                                                 #used to count tokens
import tensorflow as tf                                                         #model backend
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Flatten, Activation, dot, concatenate, multiply

from pipe import traverse                                                       #useful
import numpy as np                                                              #for messing with tensors
import gc                                                                       #keeps memory usage low
from tensorflow.keras import optimizers                                         #adam

maxPairs = 15000
USE_FREQUENCY_RESTRICTION = False
latent_dim = 128
epochs = 5000
batch_size = 128

tf.config.list_physical_devices('GPU')

global data
dataRaw = ParallelCorpus("eng","tok")

def process_sentence(s):
    s = decode(s.lower())
    s = sub(r'([!.?])', r' \1', s)
    s = sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = sub(r'\s+', r' ', s)
    s = s.strip()
    s = '<s>' +' '+ s +' '+'</s>'
    return s

data=[]
for s,t in dataRaw:                                                             #no length check here as actual number of pairs is very small
  data.append((process_sentence(s.text),process_sentence(t.text)))

def shuffleData():
  global data
  shuffle(data)
shuffleData();                                                                  #ensures that the data used in the first few epochs is always different
print("Example pair: %s" % str(data[0])[1:-1].replace("', ", "' --> '"))        #prints an example bitext pair

rawEn,rawTp = list(zip(*data));english,toki = list(rawEn),list(rawTp)
english,toki=list(map(lambda x:x.split(" "),english)),list(map(lambda x:x.split(" "),toki)) #ace what the fuck

print(len(english))
print(max(list(map(len, english))), max(list(map(len, toki))))

#i cant lie i love this little bit of code so much
zipped = sorted(list(zip(english,toki)),key=lambda x:len(x[1]))
(english, toki) = zip(*zipped[:round(len(zipped)*99/100)]) #strip longest hundreth
del zipped

print(len(english))
print(max(list(map(len, english))), max(list(map(len, toki))))

flattened_english = list(english | traverse)
flattened_toki = list(toki | traverse)
num_pairs=len(english)
english_counter=Counter(flattened_english)
toki_counter=Counter(flattened_toki)

eng_words = list(english_counter.keys())
tok_words = list(toki_counter.keys())

toki_tokenizer=dict(zip(sorted(tok_words),list(range(0, len(tok_words)))))
english_tokenizer=dict(zip(sorted(eng_words),list(range(0, len(eng_words)))))

max_english_sentence_length=max(list(map(len, english)))
max_toki_sentence_length=max(list(map(len, toki)))

num_encoder_tokens=len(english_tokenizer)
num_decoder_tokens=len(tok_words)+USE_FREQUENCY_RESTRICTION

#data stuff

encoder_input_data = np.ndarray((maxPairs,max_english_sentence_length))
gc.collect()

i=0
for seq in english[:maxPairs]:
  temp = list(map(lambda x:english_tokenizer[x], seq))
  zeros = [0]*(max_english_sentence_length - len(temp))
  encoder_input_data[i] = np.array(temp+zeros)
  i+=1
decoder_input_data = np.ndarray((maxPairs,max_toki_sentence_length))
gc.collect()

i=0
for seq in toki[:maxPairs]:
  temp = list(map(lambda x:toki_tokenizer[x], seq))
  zeros = [0]*(max_toki_sentence_length - len(temp))
  decoder_input_data[i] = np.array(temp+zeros)
  i+=1
def onehot(seq):
  seq = list(map(lambda x:toki_tokenizer[x], seq))
  out = np.zeros((max_toki_sentence_length,num_decoder_tokens))
  i=0
  for token in seq:
    temp = np.zeros(num_decoder_tokens)
    temp[token]=1.0
    out[i]=temp;i+=1
  return out
decoder_target_data = np.ndarray((maxPairs,max_toki_sentence_length,num_decoder_tokens))
gc.collect()

i=0
for each in toki[:maxPairs]:
  seq=list(each[1:])
  decoder_target_data[i] = onehot(seq)
  i+=1

#This is an offset check, the second line (target data) should be offset by a single timestep.
print(str(list(map(lambda x:f'{int(x):03}', decoder_input_data[5]))).replace("'",""))
def f(x):
  try: return f'{int(x.index(1.0)):03}'
  except: return '000'
print(str(list(map(lambda x:f(list(x)),list(decoder_target_data[5])))).replace("'",""))

#actual model fitting
gc.collect()
from tensorflow.keras import optimizers

#encoder embedding and input layers
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, latent_dim,input_length=max_english_sentence_length)(encoder_inputs)
encoder_stack_h, encoder_state_h, encoder_state_c = LSTM(latent_dim,return_state=True,return_sequences=True)(encoder_embedding)
encoder_states = [encoder_state_h, encoder_state_c]

#decoder embedding and dense layer (STOP SETTING THE DENSE NEURON COUNT TO ONE ACE)
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, latent_dim,input_length=max_toki_sentence_length)(decoder_inputs)
decoder_stack_h = LSTM(latent_dim, return_sequences=True)(decoder_embedding, initial_state=encoder_states)

attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])                #finds the dot product between the sequences
attention = Activation('softmax')(attention)                                    #softmaxes it
context = dot([attention, encoder_stack_h], axes=[2,1])                         #calculates the context vectors
decoder_combined_context = concatenate([context, decoder_stack_h])              #combines the context and hidden states


decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(decoder_combined_context) #then finds which word is most likely

#compile the model and optimizer
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=["accuracy"])

#just summary things
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/",
    verbose=1,
    save_weights_only=False,
    save_freq='epoch',period=500)

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[cp_callback]
          )
