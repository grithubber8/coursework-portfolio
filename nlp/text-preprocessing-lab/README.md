# Text Preprocessing Lab (TP2)

A lab on **text normalisation**, the first stage of any NLP pipeline. It takes a raw paragraph and cleans it step by step, then answers short conceptual questions about why each step matters.

**What it covers**
- Lowercasing, and why it keeps "Python" and "python" as one word
- Removing numbers and punctuation with regular expressions (`re.sub`)
- Whitespace clean-up
- Tokenisation and stop-word removal with spaCy
- Storing the result as a token list (the input format that vectorisers and embeddings expect)
- spaCy's built-in token attributes (`is_stop`, `is_punct`, `is_digit`) as a cleaner alternative to regular expressions

**Files:** `tp2_text_preprocessing.ipynb`

**Tools:** Python, spaCy, `re`

This is a short course exercise on one example paragraph, not a benchmark. The full pipeline applied to a real dataset is in [`../airline-sentiment`](../airline-sentiment).

*Course lab for Introduction to NLP, École Centrale Casablanca, by Eunice Agogo.*
