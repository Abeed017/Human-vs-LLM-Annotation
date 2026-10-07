# Human vs LLM Annotation Pipeline

A multilingual NLP annotation pipeline that compares Human-generated labels and LLM-generated annotations for low-resource NLP tasks.

## Features

- Sentiment Classification
- Emotion Classification
- Topic Classification
- Human vs AI annotation comparison
- English, Kannada, Indonesian support
- Accuracy
- Precision
- Recall
- F1 Score
- Kappa Score
- Streamlit Dashboard
- CSV Upload support

## Project Structure

nlp_annotation_pipeline/

├── data/

│   ├── raw/

│   ├── processed/

│   └── annotations/

├── src/

│   ├── data_collection.py

│   ├── preprocessor.py

│   ├── llm_annotator.py

│   ├── evaluator.py

│   └── visualizer.py

├── guidelines/

├── outputs/

├── requirements.txt

├── streamlit_app.py

└── flask_app.py


## Installation

```bash
python -m pip install -r requirements.txt