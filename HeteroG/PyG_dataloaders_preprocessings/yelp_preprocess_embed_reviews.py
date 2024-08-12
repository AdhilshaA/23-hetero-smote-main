import numpy as np 
import json
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# for making review embeddings; takes a long time (~2-3 hours minimum), only run if necessary.

data_dir = '/home/deependra/project/23-hetero-smote/HeteroG/data/yelp_kaggle/'
review_file = 'yelp_academic_dataset_review.json'
review_ids_file = 'review_ids.txt'
load_full_model = False # previosuly trained word2vec model on all reviews
load_small_model = False # previously trained word2vec model on subset reviews made

print('Loading review data..')
review_data = []
with open(data_dir + review_file, encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        review_data.append(list(json.loads(line).values()))
    review_header = dict(zip(json.loads(line).keys(), range(len(json.loads(line).keys()))))

review_ids = {}
print('Loading review ids..')
with open(data_dir + 'review_ids.txt', 'r') as f:
    for line in tqdm(f.readlines()):
        review_ids[line.strip().split('\t')[1]] = int(line.strip().split('\t')[0])

print('Extracting needed indices for reviews..')
id_dataidx = {}
for i in tqdm(range(len(review_data))):
    if review_data[i][review_header['review_id']] in review_ids:
        id_dataidx[review_data[i][review_header['review_id']]] = i

# creating review embeddings using word2vec
print('Tokenizing reviews..')
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    sentence = word_tokenize(text)
    sentence = [word for word in sentence if word not in stop_words]
    return sentence

print('Extracting sentences + preprocessing..')
sentences = []
for name,id in tqdm(review_ids.items()):
    sentences.append(preprocess(review_data[id_dataidx[name]][review_header['text']]))

print(f'Number of selected reviews: {len(review_ids)}')
print(f'Number of sentences: {len(sentences)}')


if load_full_model:
    print(f'loading KeyedVectors from {data_dir+"full_reviews_word2vec.txt"}..')
    keyed_vectors = KeyedVectors.load_word2vec_format(data_dir+"full_reviews_word2vec.txt",binary=False)
elif load_small_model:
    print(f'loading KeyedVectors from {data_dir+"small_reviews_word2vec.txt"}..')
    keyed_vectors = KeyedVectors.load_word2vec_format(data_dir+"small_reviews_word2vec.txt",binary=False)
else:
    # Train a Word2Vec model
    print('training word2vec model..')
    model = Word2Vec(sentences=sentences, vector_size=128, window=5, min_count=1, sg=1, negative=5,)
    keyed_vectors = model.wv
    
    print(f'saving keyed vectors to {data_dir+"small_reviews_word2vec.txt"}..')
    keyed_vectors.save_word2vec_format(data_dir+"small_reviews_word2vec.txt", binary=False)
    del model

# Get embeddings for each review
print('getting embeddings for each review..')
count = 0
review_embeddings = []
for sentence in tqdm(sentences):
    # Calculate the average vector for each word in a review
    sentence_vectors = [keyed_vectors[word] for word in sentence if word in keyed_vectors]
    if sentence_vectors:
        average_vector = sum(sentence_vectors) / len(sentence_vectors)
        review_embeddings.append(average_vector)
    else:
        # Handle cases where none of the sentence are in the vocabulary
        review_embeddings.append([0]*128)
        count += 1
print(f"Number of reviews with no embeddings: {count}")

print('saving review embeddings..')
review_embeddings = np.array(review_embeddings)

print(f'saving review embeddings to {data_dir+"review_embeddings.npy"}..')
np.save(data_dir+"review_embeddings.npy", review_embeddings)