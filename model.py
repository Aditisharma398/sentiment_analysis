from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from preprocess import load_data

# load data
df = load_data()

X = df['text']
y = df['label']

# convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# train model
model = LogisticRegression()
model.fit(X, y)

# save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")