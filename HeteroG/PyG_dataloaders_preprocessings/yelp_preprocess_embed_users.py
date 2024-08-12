import numpy as np 
import json
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# for making review embeddings; takes a long time (~2-3 hours minimum), only run if necessary.

data_dir = '/home/deependra/project/23-hetero-smote/HeteroG/data/yelp_kaggle/'
user_file = 'yelp_academic_dataset_user.json'
user_ids_file = 'user_ids.txt'
load_small_model = False # previously trained word2vec model on subset users made

u_columns = ['name', 'yelping_since', 'useful', 'funny', 'cool', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos']

print('Loading user data..')
user_data = []
with open(data_dir + user_file, encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        user_data.append(list(json.loads(line).values()))
    user_header = dict(zip(json.loads(line).keys(), range(len(json.loads(line).keys()))))

u_columns_idx = [user_header[col] for col in u_columns]

user_ids = {}
print('Loading user_ids..')
with open(data_dir + 'user_ids.txt', 'r') as f:
    for line in tqdm(f.readlines()):
        user_ids[line.strip().split('\t')[1]] = int(line.strip().split('\t')[0])

print('Extracting needed indices for users..')
id_dataidx = {}
for i in tqdm(range(len(user_data))):
    if user_data[i][user_header['user_id']] in user_ids:
        id_dataidx[user_data[i][user_header['user_id']]] = i

# creating user embeddings using word2vec
print('Tokenizing users..')
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    sentence = word_tokenize(text)
    sentence = [word for word in sentence if word not in stop_words]
    return sentence

print('Extracting sentences..')
sentences = []
for name,id in tqdm(user_ids.items()):
    sentences.append(' '.join([str(user_data[id_dataidx[name]][idx]) for idx in u_columns_idx]))

print(f'Number of selected users: {len(user_ids)}')
print(f'Number of sentences: {len(sentences)}')

print('Preprocessing sentences..')
for i in tqdm(range(len(sentences))):
    sentences[i] = preprocess(sentences[i])

if load_small_model:
    print(f'loading KeyedVectors from {data_dir+"small_users_word2vec.txt"}..')
    keyed_vectors = KeyedVectors.load_word2vec_format(data_dir+"small_users_word2vec.txt",binary=False)
else:
    # Train a Word2Vec model
    print('training word2vec model..')
    model = Word2Vec(sentences=sentences, vector_size=128, window=5, min_count=1, sg=1, negative=5,)
    keyed_vectors = model.wv
    
    print(f'saving keyed vectors to {data_dir+"small_users_word2vec.txt"}..')
    keyed_vectors.save_word2vec_format(data_dir+"small_users_word2vec.txt", binary=False)
    del model

# Get embeddings for each user
print('getting embeddings for each user..')
count = 0
user_embeddings = []
for sentence in tqdm(sentences):
    # Calculate the average vector for each word in a user
    sentence_vectors = [keyed_vectors[word] for word in sentence if word in keyed_vectors]
    if sentence_vectors:
        average_vector = sum(sentence_vectors) / len(sentence_vectors)
        user_embeddings.append(average_vector)
    else:
        # Handle cases where none of the sentence are in the vocabulary
        user_embeddings.append([0]*128)
        count += 1
print(f"Number of users with no embeddings: {count}")

print('saving user embeddings..')
user_embeddings = np.array(user_embeddings)

print(f'saving user embeddings to {data_dir+"user_embeddings.npy"}..')
np.save(data_dir+"user_embeddings.npy", user_embeddings)