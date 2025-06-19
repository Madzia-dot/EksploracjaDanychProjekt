import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# 1. Wczytanie i eksploracja danych

df = pd.read_csv("data/train.csv")

print("Liczba rekordów:", len(df))
print("\nPodgląd danych:")
print(df.head())

print("\n Brakujące dane:")
print(df.isnull().sum())

duplikaty = df.duplicated().sum()
print(f"\nLiczba duplikatów: {duplikaty}")

df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))
print("\nStatystyki długości tekstów:")
print(df["text_length"].describe())

plt.figure(figsize=(8, 4))
sns.histplot(df["text_length"], bins=50, kde=True)
plt.title("Rozkład długości tekstów (w słowach)")
plt.xlabel("Liczba słów")
plt.ylabel("Liczba wiadomości")
plt.tight_layout()
plt.show()

print("\nRozkład kategorii:")
print(df["Category"].value_counts())

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="Category", order=df["Category"].value_counts().index)
plt.title("Liczba wiadomości w każdej kategorii")
plt.xlabel("Kategoria")
plt.ylabel("Liczba wiadomości")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n Najkrótsze teksty:")
print(df[df["text_length"] < 5][["text", "Category"]].head())

def zawiera_link(text):
    return bool(re.search(r"http[s]?://", str(text)))

df["has_link"] = df["text"].apply(zawiera_link)
print("\n Liczba tekstów z linkami:", df["has_link"].sum())


# 2. Czyszczenie danych

df = df.dropna(subset=["text"])
df = df.drop_duplicates()

print("\n Po czyszczeniu:", df.shape)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# 3. Podział na zbiory
X = df["clean_text"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Wektoryzacja tekstu
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Budowa i trenowanie modeli

models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTrenowanie modelu: {name}")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} - Accuracy: {acc:.4f}")
    print(f"\n{name} - Raport klasyfikacji:\n{classification_report(y_test, y_pred)}")

    results[name] = {
        "model": model,
        "accuracy": acc
    }


# 7. Zapis najlepszego modelu

best_model_name = max(results, key=lambda name: results[name]["accuracy"])
best_model = results[best_model_name]["model"]

joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print(f"\n Najlepszy model ({best_model_name}) i wektoryzator zapisane jako model.pkl i vectorizer.pkl")


# 9. Porównanie dokładności modeli – wykres
model_names = list(results.keys())
accuracies = [results[name]["accuracy"] for name in model_names]

plt.figure(figsize=(8, 5))
sns.barplot(x=accuracies, y=model_names, palette="viridis")
plt.xlabel("Dokładność (Accuracy)")
plt.title("Porównanie dokładności modeli")
plt.xlim(0, 1)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
for i, acc in enumerate(accuracies):
    plt.text(acc + 0.01, i, f"{acc:.2f}", va="center")
plt.tight_layout()
plt.show()




