import json
from jsonvectorizer import *
from jsonvectorizer.utils import *
import pickle
import scipy

# Load data
docs = []
i=1
with utils.fopen(r"C:\Users\11818\Desktop\RL\Code\vae\test_data\new.json") as f:
    for line in f:
        doc = json.loads(line)
        docs.append(doc)
        print(i)
        i=i+1

docs.append(json.load(open(r"C:\Users\11818\Desktop\RL\Code\vae\test_data\1.1.1.1.json")))
# Learn the schema of sample documents
vectorizer = JsonVectorizer()
vectorizer.extend(docs)
vectorizer.prune(patterns=['^_'], min_f=0.01)

# Report booleans as is
bool_vectorizer = {
    'type': 'boolean',
    'vectorizer': vectorizers.BoolVectorizer
}

# For numbers, use one-hot encoding with 10 bins
number_vectorizer = {
    'type': 'number',
    'vectorizer': vectorizers.NumberVectorizer,
    'kwargs': {'n_bins': 10},
}

# For strings use tokenization, ignoring sparse (<1%) tokens
string_vectorizer = {
    'type': 'string',
    'vectorizer': vectorizers.StringVectorizer,
    'kwargs': {'min_df': 0.01}
}

# Build JSON vectorizer
vectorizers = [
    bool_vectorizer,
    number_vectorizer,
    string_vectorizer
]
vectorizer.fit(vectorizers=vectorizers)


for i, feature_name in enumerate(vectorizer.feature_names_):
    print('{}: {}'.format(i, feature_name))

# Convert to CSR format for efficient row slicing

X = vectorizer.transform(docs).tocsr()

scipy.sparse.save_npz('data\\CSR_test.npz', X)

with open('data\\featrue_test.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
