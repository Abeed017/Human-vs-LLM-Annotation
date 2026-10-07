import os
import pandas as pd
import streamlit as st

from src.data_collection import load_dataset
from src.preprocessor import build_dataset
from src.llm_annotator import (
    annotate_dataset_with_llm,
    annotate_dataset_with_classifier,
    LABEL_OPTIONS
)

from src.evaluator import evaluate_scheme

from src.visualizer import (
    plot_confusion_matrix,
    plot_metrics
)

# Sample datasets
SAMPLE_DATASETS={

"Indonesian sentiment":
os.path.join(
"data","raw",
"sample_texts.csv"
),

"Emotion dataset":
os.path.join(
"data","raw",
"emotion_dataset.csv"
),

"Topic dataset":
os.path.join(
"data","raw",
"topic_dataset.csv"
)
}


def main():

    st.set_page_config(
        page_title="NLP Annotation Pipeline",
        layout="wide"
    )

    st.title("Human vs LLM Annotation Pipeline")

    st.markdown("""
    This system compares human-generated labels
    with AI-generated annotations using NLP models.
    """)

    # Sidebar
    with st.sidebar:

        st.header("Configuration")

        task = st.selectbox(
            "Task",
            list(LABEL_OPTIONS.keys()),
            index=0
        )

        engine = st.selectbox(
            "LLM Engine",
            ["none", "openai", "anthropic"],
            index=0
        )

        classifier_model = st.text_input(
            "Classifier model",
            "indobenchmark/indobert-base-p1"
        )

        language = st.text_input(
            "Language",
            "Indonesian"
        )

        data_source = st.radio(
            "Dataset Source",
            ["Sample data", "Upload CSV"],
            index=0
        )

        sample_dataset = st.selectbox(
            "Sample Dataset",
            list(SAMPLE_DATASETS.keys()),
            index=0
        )

        st.markdown("---")

        st.write("Environment Variables")

        st.code("""
OPENAI_API_KEY
ANTHROPIC_API_KEY
""")

    # Load dataset
    if data_source == "Upload CSV":

        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"]
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = None

    else:

        sample_path = SAMPLE_DATASETS.get(
            sample_dataset,
            list(SAMPLE_DATASETS.values())[0]
        )

        df = load_dataset(sample_path)

    # Validation
    if df is None:
        st.warning(
            "Upload a CSV file or choose sample dataset."
        )
        return

    if "text" not in df.columns:
        st.error(
            "Dataset must contain a 'text' column."
        )
        return

    # Dataset preview
    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.write(f"Loaded {len(df)} rows.")

    # Preprocessing
    if "df_proc" not in st.session_state:

        st.session_state.df_proc = build_dataset(
            df,
            task=task
        )

    if st.button("Preprocess Dataset"):

        st.session_state.df_proc = build_dataset(
            df,
            task=task
        )

        st.success("Preprocessing complete.")

    df_proc = st.session_state.df_proc

    st.subheader("Preprocessed Data")

    st.dataframe(df_proc.head())

    st.write(f"Processed {len(df_proc)} rows.")

    # Annotation buttons
    col1, col2 = st.columns(2)

    # Classifier annotation
    with col1:

        if st.button("Annotate with Classifier"):

            try:

                st.session_state.df_proc = annotate_dataset_with_classifier(
                    df_proc,
                    task=task,
                    language=language,
                    model_name=classifier_model
                )

                st.success(
                    "Classifier annotation complete."
                )

            except Exception as exc:

                st.error(
                    f"Classifier annotation failed: {exc}"
                )

    # LLM annotation
    with col2:

        if st.button("Annotate with LLM"):

            if engine == "none":

                st.warning(
                    "Select LLM engine first."
                )

            else:

                try:

                    st.session_state.df_proc = annotate_dataset_with_llm(
                        df_proc,
                        task=task,
                        language=language,
                        engine=engine
                    )

                    st.success(
                        "LLM annotation complete."
                    )

                except Exception as exc:

                    st.error(
                        f"LLM annotation failed: {exc}"
                    )

    # Refresh dataframe
    df_proc = st.session_state.df_proc

    # Show annotated data
    st.subheader("Annotated Dataset")

    st.dataframe(df_proc.head())

    # Evaluation
    if "gold" in df_proc.columns:

        if st.button("Evaluate Annotations"):

            st.subheader("Evaluation Results")

            # LLM Evaluation
            if "llm_label" in df_proc.columns:

                st.markdown("## LLM Results")

                llm_results = evaluate_scheme(
                    df_proc["gold"],
                    df_proc["llm_label"]
                )

                st.write(
                    f"Accuracy: {llm_results['accuracy']:.2f}"
                )

                st.write(
                    f"Precision: {llm_results['precision']:.2f}"
                )

                st.write(
                    f"Recall: {llm_results['recall']:.2f}"
                )

                st.write(
                    f"F1 Score: {llm_results['f1_score']:.2f}"
                )
                st.write(f"Kappa Score:{llm_results['kappa']:.2f}"
                )
                # Metrics graph
                fig_metrics = plot_metrics(
                    llm_results
                )

                st.pyplot(fig_metrics)

                # Confusion matrix
                labels = list(
                    df_proc["gold"].unique()
                )

                fig_cm = plot_confusion_matrix(
                    llm_results["confusion_matrix"],
                    labels
                )

                st.pyplot(fig_cm)

                # Classification report
                report_df = pd.DataFrame(
                    llm_results["classification_report"]
                ).transpose()

                st.subheader("LLM Classification Report")

                st.dataframe(report_df)

            # BERT Evaluation
            if "bert_label" in df_proc.columns:

                st.markdown("## BERT Results")

                bert_results = evaluate_scheme(
                    df_proc["gold"],
                    df_proc["bert_label"]
                )

                st.write(
                    f"Accuracy: {bert_results['accuracy']:.2f}"
                )

                st.write(
                    f"Precision: {bert_results['precision']:.2f}"
                )

                st.write(
                    f"Recall: {bert_results['recall']:.2f}"
                )

                st.write(
                    f"F1 Score: {bert_results['f1_score']:.2f}"
                )

                # Metrics graph
                fig_metrics = plot_metrics(
                    bert_results
                )

                st.pyplot(fig_metrics)

                # Confusion matrix
                labels = list(
                    df_proc["gold"].unique()
                )

                fig_cm = plot_confusion_matrix(
                    bert_results["confusion_matrix"],
                    labels
                )

                st.pyplot(fig_cm)

                # Classification report
                report_df = pd.DataFrame(
                    bert_results["classification_report"]
                ).transpose()

                st.subheader("BERT Classification Report")

                st.dataframe(report_df)

    else:

        st.info(
            "Dataset must contain a 'gold' column for evaluation."
        )

    # Download CSV
    st.subheader("Download Results")

    csv = df_proc.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Annotated CSV",
        data=csv,
        file_name="annotated_dataset.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()