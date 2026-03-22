import pandas as pd
import numpy as np
import re
from datetime import datetime

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
print(f"Loaded {len(df)} medical notes")
print("\nFirst 5 notes:")
print(df.head())


def question_one():
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4}'

    results = []
    for note in df:
        matches = re.findall(pattern, note)
        results.extend(matches)

    return results

q1_result = question_one()
print(f"Found {len(q1_result)} dates")
print(f"First 10: {q1_result[:10]}")


def question_two():
    pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*-?\d{1,2}[a-z]*,?\s*\d{4}'

    results = []
    for note in df:
        matches = re.findall(pattern, note)
        results.extend(matches)

    return results

q2_result = question_two()
print(f"Found {len(q2_result)} dates")
print(f"First 10: {q2_result[:10]}")


def question_three():
    pattern = r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s\d{4}'

    results = []
    for note in df:
        matches = re.findall(pattern, note)
        results.extend(matches)

    return results

q3_result = question_three()
print(f"Found {len(q3_result)} dates")
print(f"First 10: {q3_result[:10]}")


def question_four(text):
    pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.]+'

    return re.findall(pattern, text)

test_text = """
Contact us at support@company.com or sales@company.org.
You can also reach john.doe@email.co.uk or jane_doe123@university.edu.
Invalid emails: @invalid.com, user@, not-an-email
"""

q4_result = question_four(test_text)
print(f"Found emails: {q4_result}")


def question_five(text):
    text = re.sub(r'\d', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    return text

test_text = "Hello, World! 123 This is a TEST... with 456 numbers!!!"
q5_result = question_five(test_text)
print(f"Original: '{test_text}'")
print(f"Cleaned:  '{q5_result}'")


def question_six(text):
    pattern = r'\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}'

    matches = re.findall(pattern, text)

    standardized = []
    for match in matches:
        digits = re.sub(r'\D', '', match)
        if len(digits) == 10:
            standardized.append(f"{digits[:3]}-{digits[3:6]}-{digits[6:]}")

    return standardized

test_text = """
Call us at 123-456-7890 or (555) 123-4567.
You can also reach us at 888.555.1234 or 999 888 7777.
Invalid: 12-34-5678, 1234567890
"""

q6_result = question_six(test_text)
print(f"Found phones: {q6_result}")


def question_seven():
    dates = {}

    for i, note in enumerate(df):

        # MM/DD/YY or MM/DD/YYYY
        m = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', note)
        if m:
            month, day, year = m.group(1), m.group(2), m.group(3)
            if len(year) == 2:
                year = '19' + year
            dates[i] = datetime(int(year), int(month), int(day))
            continue

        # Month name DD, YYYY or Month DD YYYY etc.
        m = re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*-?(\d{1,2})[a-z]*,?\s*(\d{4})', note)
        if m:
            month_str = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', note).group(1)
            day, year = m.group(1), m.group(2)
            dates[i] = datetime.strptime(f"{month_str} {day} {year}", "%b %d %Y")
            continue

        # DD Month YYYY
        m = re.search(r'(\d{1,2})\s((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\.?\s(\d{4})', note)
        if m:
            day, month_str, year = m.group(1), m.group(2)[:3], m.group(3)
            dates[i] = datetime.strptime(f"{month_str} {day} {year}", "%b %d %Y")
            continue

        # Month YYYY (no day)
        m = re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s(\d{4})', note)
        if m:
            month_str = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', note).group(1)
            year = m.group(1)
            dates[i] = datetime.strptime(f"{month_str} 1 {year}", "%b %d %Y")
            continue

        # MM/YYYY (no day)
        m = re.search(r'(\d{1,2})/(\d{4})', note)
        if m:
            month, year = m.group(1), m.group(2)
            dates[i] = datetime(int(year), int(month), 1)
            continue

        # YYYY only
        m = re.search(r'\b(\d{4})\b', note)
        if m:
            year = m.group(1)
            dates[i] = datetime(int(year), 1, 1)
            continue

    sorted_indices = sorted(dates, key=lambda x: dates[x])

    return pd.Series(sorted_indices)

q7_result = question_seven()
print(f"Result length: {len(q7_result)}")
print(f"First 10 indices: {list(q7_result.head(10))}")
print(f"Last 10 indices: {list(q7_result.tail(10))}")


# ── Verification ──────────────────────────────────────────────────────────

print("Checking functions...")

try:
    r1 = question_one()
    assert isinstance(r1, list), "question_one should return a list"
    print("✓ question_one: OK")
except Exception as e:
    print(f"✗ question_one: {e}")

try:
    r2 = question_two()
    assert isinstance(r2, list), "question_two should return a list"
    print("✓ question_two: OK")
except Exception as e:
    print(f"✗ question_two: {e}")

try:
    r3 = question_three()
    assert isinstance(r3, list), "question_three should return a list"
    print("✓ question_three: OK")
except Exception as e:
    print(f"✗ question_three: {e}")

try:
    r4 = question_four("test@email.com")
    assert isinstance(r4, list), "question_four should return a list"
    print("✓ question_four: OK")
except Exception as e:
    print(f"✗ question_four: {e}")

try:
    r5 = question_five("Hello World 123")
    assert isinstance(r5, str), "question_five should return a string"
    print("✓ question_five: OK")
except Exception as e:
    print(f"✗ question_five: {e}")

try:
    r6 = question_six("123-456-7890")
    assert isinstance(r6, list), "question_six should return a list"
    print("✓ question_six: OK")
except Exception as e:
    print(f"✗ question_six: {e}")

try:
    r7 = question_seven()
    assert isinstance(r7, pd.Series), "question_seven should return a pandas Series"
    print("✓ question_seven: OK")
except Exception as e:
    print(f"✗ question_seven: {e}")

print("\nDone!")
