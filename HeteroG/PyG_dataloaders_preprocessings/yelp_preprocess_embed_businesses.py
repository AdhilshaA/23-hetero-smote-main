import numpy as np 
import json
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# for making review embeddings; takes a long time (~2-3 hours minimum), only run if necessary.

data_dir = '/home/deependra/project/23-hetero-smote/HeteroG/data/yelp_kaggle/'
business_file = 'yelp_academic_dataset_business.json'
business_ids_file = 'business_ids.txt'
load_small_model = False # previously trained word2vec model on subset reviews made

b_columns = ['name', 'address', 'city', 'state', 'latitude', 'longitude', 'stars', 'categories']

print('Loading business data..')
business_data = []
with open(data_dir + business_file, encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        business_data.append(list(json.loads(line).values()))
    business_header = dict(zip(json.loads(line).keys(), range(len(json.loads(line).keys()))))

b_columns_idx = [business_header[col] for col in b_columns]

business_ids = {}
print('Loading business_ids..')
with open(data_dir + 'business_ids.txt', 'r') as f:
    for line in tqdm(f.readlines()):
        business_ids[line.strip().split('\t')[1]] = int(line.strip().split('\t')[0])

print('Extracting needed indices for reviews..')
id_dataidx = {}
for i in tqdm(range(len(business_data))):
    if business_data[i][business_header['business_id']] in business_ids:
        id_dataidx[business_data[i][business_header['business_id']]] = i

# creating review embeddings using word2vec
print('Tokenizing reviews..')
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    sentence = word_tokenize(text)
    sentence = [word for word in sentence if word not in stop_words]
    return sentence

print('Extracting sentences..')
sentences = []
for name,id in tqdm(business_ids.items()):
    sentences.append(' '.join([str(business_data[id_dataidx[name]][idx]) for idx in b_columns_idx]))

print(f'Number of selected businesses: {len(business_ids)}')
print(f'Number of sentences: {len(sentences)}')

print('Preprocessing sentences..')
for i in tqdm(range(len(sentences))):
    sentences[i] = preprocess(sentences[i])

if load_small_model:
    print(f'loading KeyedVectors from {data_dir+"small_businesses_word2vec.txt"}..')
    keyed_vectors = KeyedVectors.load_word2vec_format(data_dir+"small_businesses_word2vec.txt",binary=False)
else:
    # Train a Word2Vec model
    print('training word2vec model..')
    model = Word2Vec(sentences=sentences, vector_size=128, window=5, min_count=1, sg=1, negative=5,)
    keyed_vectors = model.wv
    
    print(f'saving keyed vectors to {data_dir+"small_businesses_word2vec.txt"}..')
    keyed_vectors.save_word2vec_format(data_dir+"small_businesses_word2vec.txt", binary=False)
    del model

# Get embeddings for each business
print('getting embeddings for each business..')
count = 0
business_embeddings = []
for sentence in tqdm(sentences):
    # Calculate the average vector for each word in a business
    sentence_vectors = [keyed_vectors[word] for word in sentence if word in keyed_vectors]
    if sentence_vectors:
        average_vector = sum(sentence_vectors) / len(sentence_vectors)
        business_embeddings.append(average_vector)
    else:
        # Handle cases where none of the sentence are in the vocabulary
        business_embeddings.append([0]*128)
        count += 1
print(f"Number of businesses with no embeddings: {count}")

print('saving business embeddings..')
business_embeddings = np.array(business_embeddings)

print(f'saving business embeddings to {data_dir+"business_embeddings.npy"}..')
np.save(data_dir+"business_embeddings.npy", business_embeddings)