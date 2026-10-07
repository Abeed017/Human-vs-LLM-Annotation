import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
from src.data_collection import load_dataset
from src.preprocessor import build_dataset
from src.llm_annotator import annotate_dataset_with_llm, annotate_dataset_with_classifier, LABEL_OPTIONS
from src.evaluator import evaluate_scheme
from src.visualizer import plot_main_effects

ALLOWED_EXTENSIONS = {"csv"}
UPLOAD_FOLDER = "uploads"
OUTPUT_IMAGE = "outputs/main_effects.png"
OUTPUT_CSV = "outputs/annotated_dataset.csv"

SAMPLE_DATASETS = {
    "Indonesian sentiment": {
        "path": os.path.join("data", "raw", "sample_texts.csv"),
        "task": "sentiment",
        "language": "Indonesian",
    },
    "Turkish sentiment": {
        "path": os.path.join("data", "raw", "sample_texts_turkish.csv"),
        "task": "sentiment",
        "language": "Turkish",
    },
}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "secret-key")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    sample_dataset = "Indonesian sentiment"
    dataset_source = "sample"
    result = None
    preview = None
    error = None
    selected_task = "sentiment"
    engine = "none"
    classifier_model = "indobenchmark/indobert-base-p1"
    language = "Indonesian"

    if request.method == "POST":
        sample_dataset = request.form.get("sample_dataset", sample_dataset)
        selected_task = request.form.get("task", "sentiment")
        engine = request.form.get("engine", "none")
        classifier_model = request.form.get("classifier_model", classifier_model)
        language = request.form.get("language", language)
        dataset_source = request.form.get("dataset_source", "sample")

        if dataset_source == "sample":
            sample_info = SAMPLE_DATASETS.get(sample_dataset, SAMPLE_DATASETS["Indonesian sentiment"])
            df = load_dataset(sample_info["path"])
            if not request.form.get("language"):
                language = sample_info["language"]
            if not request.form.get("task"):
                selected_task = sample_info["task"]
        else:
            file = request.files.get("file")
            if not file or file.filename == "":
                flash("Please choose a CSV file to upload.")
                return redirect(request.url)
            if not allowed_file(file.filename):
                flash("Only CSV files are supported.")
                return redirect(request.url)
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            df = load_dataset(file_path)

        try:
            df_proc = build_dataset(df, task=selected_task)
            preview = df_proc.head(10).to_dict(orient="records")

            if engine != "none":
                df_proc = annotate_dataset_with_llm(df_proc, task=selected_task, language=language, engine=engine)
            df_proc = annotate_dataset_with_classifier(df_proc, task=selected_task, language=language, model_name=classifier_model)

            if "gold" in df_proc.columns:
                results = []
                if "llm_label" in df_proc.columns and engine != "none":
                    results.append(evaluate_scheme(df_proc["gold"], df_proc["llm_label"], scheme_name="LLM"))
                if "bert_label" in df_proc.columns:
                    results.append(evaluate_scheme(df_proc["gold"], df_proc["bert_label"], scheme_name="BERT"))
                plot_main_effects(results)
                result = results

            df_proc.to_csv(OUTPUT_CSV, index=False)
            preview = df_proc.head(10).to_dict(orient="records")
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        tasks=LABEL_OPTIONS.keys(),
        sample_datasets=SAMPLE_DATASETS.keys(),
        selected_sample=sample_dataset,
        dataset_source=dataset_source,
        selected_task=selected_task,
        engine=engine,
        classifier_model=classifier_model,
        language=language,
        preview=preview,
        result=result,
        error=error,
        output_image=os.path.exists(OUTPUT_IMAGE),
        output_csv=os.path.exists(OUTPUT_CSV),
    )


@app.route("/outputs/main_effects.png")
def output_image():
    if os.path.exists(OUTPUT_IMAGE):
        return send_file(OUTPUT_IMAGE, mimetype="image/png")
    flash("No chart is available yet. Run annotation first.")
    return redirect(url_for("index"))


@app.route("/download")
def download_csv():
    if os.path.exists(OUTPUT_CSV):
        return send_file(OUTPUT_CSV, as_attachment=True)
    flash("No annotated dataset is available for download.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=8501)
