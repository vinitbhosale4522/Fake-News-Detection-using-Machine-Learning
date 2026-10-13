# Project Audit Notes (separate-copy review)

## Scope and preservation
- Source `.py` files, dataset CSVs, notebook, README, requirements, metrics, and serialized model files were preserved byte-for-byte from the supplied archive.
- Only `__pycache__/` Python bytecode cache files were removed from this copy because Python can regenerate them.
- No source code was edited and no model was retrained.

## File-by-file review
- `app.py`: Streamlit UI with prediction form, session-only prediction history, analytics, model metrics, and About page. Python syntax parses successfully. Imports `joblib` but it appears unused in the inspected source; several dashboard dependencies are used.
- `prediction.py`: loads `model.pkl` and `vectorizer.pkl`, preprocesses input, predicts label/probabilities, and reports coefficient-based keyword indicators. Syntax parses successfully. Its `__main__` block prints sample predictions (expected command-line demo output, not application debug logging).
- `preprocessing.py`: lowercases, removes HTML/URLs/punctuation/numbers, removes English stopwords, and applies Porter stemming. It may download NLTK data if absent. Syntax parses successfully. Its `__main__` block prints a sample transformation.
- `train_model.py`: loads CSV files, creates labels, preprocesses, fits TF-IDF and trains Logistic Regression, Naive Bayes, and Random Forest; saves model artifacts and evaluation metrics. Syntax parses successfully. Console `print()` statements are progress/reporting output, not necessarily accidental debug code.
- `generate_dataset.py`: creates synthetic, template-based sample rows and writes `Fake.csv` and `True.csv`. This is a critical provenance/validity point: these generated examples are not genuine news articles, and performance on them does not establish real-world fake-news detection ability.
- `notebooks/fake_news_detection_eda.ipynb`: EDA and modeling notebook. Contains no saved execution output in the inspected cells. Its markdown says the dataset is a general end-to-end pipeline, but the current CSVs are synthetic template data.
- `README.md`: retained unchanged; see accuracy issues below.
- `requirements.txt`: lists the primary application and ML dependencies. Python-version compatibility should be tested in a clean environment; README badges state Python 3.11+ and package lower bounds, not a fully pinned reproducible environment.
- `dataset/Fake.csv`, `dataset/True.csv`: each has 1,500 rows and columns `title`, `text`, `subject`, `date`. Content inspection shows repeated templates, synthetic phrases, generated IDs, and deliberately sensational fake examples.
- `model.pkl`, `vectorizer.pkl`, `models_comparison.pkl`: serialized artifacts. Their existence/size was verified; this audit did not execute or retrain them.
- `metrics.json`: valid JSON; reports 3,000 rows, balanced 1,500/1,500, and 100% scores for the listed models.

## README claims that need correction before public use
1. **Dataset provenance is inaccurate/misleading:** README calls the files a Kaggle Fake & Real News dataset, but the included `generate_dataset.py` and current CSV content indicate synthetic template-generated examples. Do not claim Kaggle provenance unless you replace/verify the data and include its actual source/license.
2. **100% metrics are not evidence of real-world performance:** scores are reported on a synthetic, template-based dataset. Models may learn template/style artifacts. Describe the numbers as results on the current synthetic test split, not general fake-news detection accuracy. Prefer evaluation on a verified, licensed real-world dataset with leakage checks.
3. **README says `generate_dataset.py` creates a Kaggle-style dataset:** the script creates synthetic records. Label it synthetic sample data.
4. **Explainability wording is overstated:** keyword indicators are based on Logistic Regression coefficients for individual tokens, not a full per-example explanation method such as SHAP/LIME. The code assigns positive coefficients to Real indicators and negative coefficients to Fake indicators, assuming the classifier's class ordering remains `[0, 1]`.
5. **Confidence is not fact verification:** predicted class probability is a model score and should not be presented as certainty that a news story is true or false.
6. **License claim:** README references an MIT license badge and `LICENSE`, but no `LICENSE` file exists in the archive. Add a license only if the project owner has rights to license all included code/data/assets.
7. **Repository URL is placeholder:** clone command uses `your-username`; replace it with the actual repository URL before publishing.
8. **Deployment build command retrains a model:** `python train_model.py` during deployment may be slow and overwrites artifacts; confirm this is intended before following those instructions.
9. **Feature wording:** README says “input validation” handles “non-sensical inputs”; code mainly checks empty input and input that becomes empty after preprocessing. It cannot verify factuality or reliably detect nonsensical text.
10. **No live verification performed:** this was a static archive review; dependencies were not installed and the Streamlit app was not run.

## Attribution and viva preparation
- The archive contains no explicit `ChatGPT`, `OpenAI`, `Copilot`, `Claude`, `TODO`, or `FIXME` strings in the inspected Python source. A text scan cannot establish who authored code or prove originality.
- Attribute and retain licenses for the dataset, any copied code, icon/image assets, and any external snippets. The source of the current CSVs should be stated honestly as synthetic/template-generated unless independently verified otherwise.
- Be ready to explain: TF-IDF (unigrams/bigrams and `max_features=5000`), train/test split with stratification, Logistic Regression vs Naive Bayes vs Random Forest, precision/recall/F1/confusion matrix/ROC-AUC, preprocessing trade-offs, class label mapping (`Fake=0`, `Real=1`), probability vs certainty, and why synthetic templates can cause artificially perfect scores.

## Scan results
- Python syntax parsing succeeded for `app.py`, `prediction.py`, `preprocessing.py`, `train_model.py`, and `generate_dataset.py`.
- No code modifications were made.
