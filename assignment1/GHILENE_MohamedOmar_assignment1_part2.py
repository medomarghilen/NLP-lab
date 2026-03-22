import nltk
import pandas as pd
import numpy as np
import re

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

with open('moby.txt', 'r') as f:
    moby_raw = f.read()

moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)

print(f"Loaded Moby Dick")
print(f"Raw text length: {len(moby_raw)} characters")
print(f"First 200 characters: {moby_raw[:200]}")


def example_one():
    return len(nltk.word_tokenize(moby_raw))

print(f"Total tokens: {example_one()}")

def example_two():
    return len(set(nltk.word_tokenize(moby_raw)))

print(f"Unique tokens: {example_two()}")

def example_three():
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w, 'v') for w in text1]
    return len(set(lemmatized))

print(f"Unique tokens after verb lemmatization: {example_three()}")


def question_one():
    tokens = nltk.word_tokenize(moby_raw)
    return len(set(tokens)) / len(tokens)

q1_result = question_one()
print(f"Lexical diversity: {q1_result}")


def question_two():
    tokens = nltk.word_tokenize(moby_raw)
    count = tokens.count('whale') + tokens.count('Whale')
    return (count / len(tokens)) * 100

q2_result = question_two()
print(f"Percentage of 'whale'/'Whale': {q2_result}%")


def question_three():
    fdist = nltk.FreqDist(text1)
    return fdist.most_common(20)

q3_result = question_three()
print("20 most frequent tokens:")
for token, freq in q3_result:
    print(f"  {token}: {freq}")


def question_four():
    fdist = nltk.FreqDist(text1)
    result = [word for word, freq in fdist.items() if len(word) > 5 and freq > 150]
    return sorted(result)

q4_result = question_four()
print(f"Found {len(q4_result)} tokens:")
print(q4_result)


def question_five():
    longest = max(text1, key=len)
    return (longest, len(longest))

q5_result = question_five()
print(f"Longest word: '{q5_result[0]}' with length {q5_result[1]}")


def question_six():
    fdist = nltk.FreqDist(text1)
    result = [(freq, word) for word, freq in fdist.items() if word.isalpha() and freq > 2000]
    return sorted(result, reverse=True)

q6_result = question_six()
print("Words with frequency > 2000:")
for freq, word in q6_result:
    print(f"  {word}: {freq}")


def question_seven():
    sentences = sent_tokenize(moby_raw)
    total_tokens = sum(len(word_tokenize(sent)) for sent in sentences)
    return total_tokens / len(sentences)

q7_result = question_seven()
print(f"Average tokens per sentence: {q7_result}")


def question_eight():
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in text1 if w.isalpha() and w.lower() not in stop_words]
    fdist = nltk.FreqDist(tokens)
    return fdist.most_common(10)

q8_result = question_eight()
print("10 most common words (excluding stop words):")
for word, freq in q8_result:
    print(f"  {word}: {freq}")


def question_nine():
    porter = PorterStemmer()
    stems = [porter.stem(w) for w in text1 if w.isalpha()]
    fdist = nltk.FreqDist(stems)
    return fdist.most_common(10)

q9_result = question_nine()
print("10 most common stems:")
for stem, freq in q9_result:
    print(f"  {stem}: {freq}")


def question_ten():
    text = moby_raw[:1000]
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return tokens

q10_result = question_ten()
print(f"Number of preprocessed tokens: {len(q10_result)}")
print(f"First 20 tokens: {q10_result[:20]}")


# ── Verification ──────────────────────────────────────────────────────────

print("Checking functions...")

try:
    r1 = question_one()
    assert isinstance(r1, float), "question_one should return a float"
    print("✓ question_one: OK")
except Exception as e:
    print(f"✗ question_one: {e}")

try:
    r2 = question_two()
    assert isinstance(r2, float), "question_two should return a float"
    print("✓ question_two: OK")
except Exception as e:
    print(f"✗ question_two: {e}")

try:
    r3 = question_three()
    assert isinstance(r3, list) and len(r3) == 20, "question_three should return a list of 20 tuples"
    print("✓ question_three: OK")
except Exception as e:
    print(f"✗ question_three: {e}")

try:
    r4 = question_four()
    assert isinstance(r4, list), "question_four should return a list"
    print("✓ question_four: OK")
except Exception as e:
    print(f"✗ question_four: {e}")

try:
    r5 = question_five()
    assert isinstance(r5, tuple) and len(r5) == 2, "question_five should return a tuple of 2 elements"
    print("✓ question_five: OK")
except Exception as e:
    print(f"✗ question_five: {e}")

try:
    r6 = question_six()
    assert isinstance(r6, list), "question_six should return a list"
    print("✓ question_six: OK")
except Exception as e:
    print(f"✗ question_six: {e}")

try:
    r7 = question_seven()
    assert isinstance(r7, float), "question_seven should return a float"
    print("✓ question_seven: OK")
except Exception as e:
    print(f"✗ question_seven: {e}")

try:
    r8 = question_eight()
    assert isinstance(r8, list) and len(r8) == 10, "question_eight should return a list of 10 tuples"
    print("✓ question_eight: OK")
except Exception as e:
    print(f"✗ question_eight: {e}")

try:
    r9 = question_nine()
    assert isinstance(r9, list) and len(r9) == 10, "question_nine should return a list of 10 tuples"
    print("✓ question_nine: OK")
except Exception as e:
    print(f"✗ question_nine: {e}")

try:
    r10 = question_ten()
    assert isinstance(r10, list), "question_ten should return a list"
    print("✓ question_ten: OK")
except Exception as e:
    print(f"✗ question_ten: {e}")

print("\nDone!")
