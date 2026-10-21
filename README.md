# Fake News Detection

Classifies a news article as **reliable (0)** or **fake (1)** from its headline and body text, using TF-IDF features and Logistic Regression.

**Test accuracy: 97.43%** on a held-out 20% split, with checks to make sure the model isn't cheating through data leakage.

## Results

| Metric | Score |
|---|---|
| Test accuracy | 97.43% |
| 5-fold cross-validation accuracy | 97.08% |
| Precision (fake) | 0.97 |
| Recall (fake) | 0.98 |
| F1-score (macro avg) | 0.97 |
| Test set size | 4,078 articles |

**Confusion matrix** (rows = true, columns = predicted):

|  | Predicted reliable | Predicted fake |
|---|---|---|
| **Reliable** | 2020 | 58 |
| **Fake** | 47 | 1953 |

## Dataset

- 20,800 labelled news articles (`id`, `title`, `author`, `text`, `label`), from the Kaggle Fake News dataset
- Roughly balanced: 50.1% fake
- 20,387 articles remain after removing duplicates

Place the file at `data/train.csv`.

## Pipeline

```
Headline + text
   ↓  Keep letters only, lowercase
   ↓  Remove English stopwords (NLTK)
   ↓  Porter stemming (cached with lru_cache for speed)
   ↓  TF-IDF (sublinear tf, min_df=2, max_df=0.9)
   ↓  Logistic Regression (C tuned with 5-fold GridSearchCV → best C = 100)
   ↓  Reliable / Fake + probability
```

## Avoiding data leakage

High accuracy on this dataset can be misleading, so the notebook checks three things:

1. **Author column excluded.** 98.7% of articles with no author are fake (vs 50.1% overall), so the author field would let the model "cheat". Adding it raises headline-only accuracy from 92.45% to 97.67%, which is why it's left out.
2. **Duplicates removed** before splitting, so the same article can't appear in both train and test.
3. **Publisher-name check.** The most "reliable" words include `breitbart` and `york`, because those headlines end with the publisher's name. Removing publisher names and retraining gives **96.49%**, only about one point lower, so the model is mostly learning from the article content.

## Most informative words

- **Reliable:** breitbart, mr, twitter, york, said, ms, follow, sunday, presid, saturday
- **Fake:** octob, novemb, anti, non, co, hillari, self, sourc, comment, share

## How to run

```bash
pip install pandas numpy nltk scikit-learn jupyter
jupyter notebook fake_news_detection_simple.ipynb
```

Use the prediction helper on any article:

```python
predict_news("Headline here", "Article text here")
# → 'reliable (probability of fake: 0.00)'
```

## Tech stack

Python, pandas, NumPy, NLTK, scikit-learn (TfidfVectorizer, LogisticRegression, GridSearchCV, Pipeline)

## Limitations

The articles are 2016 US political news, so the model learns that period's writing style and topics, not whether a claim is actually true. It would likely score lower on articles from publishers or time periods it hasn't seen.

## Future work

- Test on newer, out-of-domain news sources
- Compare with transformer models (e.g., fine-tuned BERT/DistilBERT)
- Deploy as a simple web app with FastAPI or Streamlit
