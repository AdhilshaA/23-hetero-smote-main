# details of imdb data used

movie_metadata.csv: 
* The original datafile from HAN paper Github repo
* 5043 movies
* 28 features
* after loading using split(',') in python, the 12th column is movie title but some lines contain ',' in the title, so the title is spread across multiple elements.
* compile them properly and load it into a numpy array. (see imdb_han_data.ipynb for details)
* remove movies with no directors or actors (see imdb_han_data.ipynb for details)


All other files are generated from movie_metadata.csv for our experiments. (see imdb_han_data.ipynb for details)

movie_id.txt: 
* 5043 movie ids (0-5042)
* id is in the exact order of movies in movie_metadata.csv

actor_id.txt:
* 6255 actor ids (0-6254) (actor1,actor2,actor3 from movie_metadata.csv compiled and enumerated)

director_id.txt:
* 2398 director ids (0-2397) (directors from movie_metadata.csv compiled and enumerated)

movie_class.txt:
* 3165 movie-class pair [0-2, (Action, Comedy, Drama)]

movie_actors.txt:
* 5043 movie-actor pair (movie to actor1, actor2, actor3 from movie_metadata.csv)

movie_directors.txt:
* 5043 movie-director pair (movie to director from movie_metadata.csv)

movie_embeddings.txt:
* 5043 movie-embedding pair (movie to 100-dim embedding from plot keywords tokenized and word2vec and averaged)

movie_keywords_word2vec.model:
* word2vec model trained on plot keywords tokenized

node_neighbors_all.txt:
* neighbors of all node types (e.g. "m0: d1590,a1063,a4273,a2730")

random_walks_all.txt:
* random walk with restart of all node types (e.g. "m0: a2730,a4273,a2730,m1660,a2730,a4273,....")

node_neighbors_topk.txt:
* top neighbors of each node all node types (e.g. "m0: m0,m2063...,m1922;d1590,d1620,...,d1025;a4273,a1063...,a3459") (node: type1nodes;type2nodes;type3nodes)



