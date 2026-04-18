from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from preprocess import load_data

df = load_data()

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(ngram_range=(1,2))  # BIG improvement
X = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=200)  # fix convergence
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained")