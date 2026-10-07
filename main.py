from src.data_collection import load_dataset
from src.preprocessor import build_dataset, train_val_test_split
from src.llm_annotator import annotate_dataset_with_llm, annotate_dataset_with_classifier
from src.evaluator import evaluate_scheme
from src.visualizer import plot_main_effects

LANGUAGE = "Indonesian"
TASK = "sentiment"


def main():
    df = load_dataset("data/raw/sample_texts.csv")
    df = build_dataset(df, task=TASK)
    train, val, test = train_val_test_split(df)

    print("Dataset shapes:", train.shape, val.shape, test.shape)
    print("\nValidation sample:")
    print(val.head(5))

    if "gold" in val.columns:
        df_val = annotate_dataset_with_llm(val.copy(), task=TASK, language=LANGUAGE, engine="openai")
        df_val = annotate_dataset_with_classifier(df_val, task=TASK, language=LANGUAGE)

        results = [evaluate_scheme(df_val["gold"], df_val["llm_label"], scheme_name="LLM")]
        if "bert_label" in df_val.columns:
            results.append(evaluate_scheme(df_val["gold"], df_val["bert_label"], scheme_name="BERT"))

        plot_main_effects(results)
    else:
        print("No 'gold' labels found in validation set. Add a 'gold' column to evaluate annotation performance.")


if __name__ == "__main__":
    main()
