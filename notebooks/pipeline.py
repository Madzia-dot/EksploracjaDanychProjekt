import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ===============================
# 🔹 1. Wczytanie i eksploracja danych
# ===============================
df = pd.read_csv("data/train.csv")

print("📁 Liczba rekordów:", len(df))
print("\n🔍 Podgląd danych:")
print(df.head())

print("\n❓ Brakujące dane:")
print(df.isnull().sum())

duplikaty = df.duplicated().sum()
print(f"\n📎 Liczba duplikatów: {duplikaty}")

df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))
print("\n📝 Statystyki długości tekstów:")
print(df["text_length"].describe())

plt.figure(figsize=(8, 4))
sns.histplot(df["text_length"], bins=50, kde=True)
plt.title("Rozkład długości tekstów (w słowach)")
plt.xlabel("Liczba słów")
plt.ylabel("Liczba wiadomości")
plt.tight_layout()
plt.show()

print("\n📊 Rozkład kategorii:")
print(df["Category"].value_counts())

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="Category", order=df["Category"].value_counts().index)
plt.title("Liczba wiadomości w każdej kategorii")
plt.xlabel("Kategoria")
plt.ylabel("Liczba wiadomości")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n⚠️ Najkrótsze teksty:")
print(df[df["text_length"] < 5][["text", "Category"]].head())

def zawiera_link(text):
    return bool(re.search(r"http[s]?://", str(text)))

df["has_link"] = df["text"].apply(zawiera_link)
print("\n🔗 Liczba tekstów z linkami:", df["has_link"].sum())

# ===============================
# 🔹 2. Czyszczenie danych
# ===============================
df = df.dropna(subset=["text"])
df = df.drop_duplicates()

print("\n✅ Po czyszczeniu:", df.shape)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

print("\n🧼 Przykład czyszczenia tekstu:")
print(df[["text", "clean_text"]].head())

# ===============================
# 🔹 3. Podział na zbiory
# ===============================
X = df["clean_text"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 🔹 4. Wektoryzacja tekstu
# ===============================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# 🔹 5. Budowa i trenowanie modelu
# ===============================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ===============================
# 🔹 6. Ewaluacja modelu
# ===============================
y_pred = model.predict(X_test_vec)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Raport klasyfikacji:\n", classification_report(y_test, y_pred))

# ===============================
# 🔹 7. Zapis modelu i wektoryzatora
# ===============================
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n💾 Model i wektoryzator zapisane jako model.pkl i vectorizer.pkl")

# ===============================
# 🔹 8. Przykład predykcji
# ===============================
example = ["Get help now! Visit http://support.com"]
example_cleaned = [clean_text(t) for t in example]
example_vec = vectorizer.transform(example_cleaned)
prediction = model.predict(example_vec)
print("\n🔮 Przykład predykcji:", prediction[0])

