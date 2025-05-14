# Instructions for Evaluating Custom Data with the Provided Embedding Evaluation Script

This guide explains how to use your own dataset with the provided Python script to evaluate embedding models for passage retrieval. The script creates vector stores from a JSON dataset and evaluates the performance of embedding models in retrieving relevant passages for given queries. Below, we detail the required data format, provide an example, and outline the necessary changes to the script to use your custom data.

## Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install matplotlib numpy tqdm langchain sentence-transformers torch chromadb
```
Additionally, ensure you have a JSON file containing your dataset in the specified format.

## Required Data Format
The JSON dataset must be a list of dictionaries, where each dictionary represents a passage with associated questions. The bare minimum required keys for each entry are:

- **annotation_id** (string): A unique identifier for the passage.
- **text** (string): The passage content to be embedded and retrieved.
- **question** (list of strings): A list of questions related to the passage. Each question is used to query the vector store.

### Optional Keys
While not required, including additional keys like `answer`, `category`, or `url` can provide context but are not used by the script for evaluation.

### Example JSON Data
Below is an example JSON file (`custom_data.json`) with the minimum required keys and some optional fields for context:

```json
[
  {
    "annotation_id": "d5f877c8-963e-4a96-925d-6e2c28417895",
    "text": "বাংলাদেশ হাইকমিশন, আবুজা, নাইজেরিয়া অফিস সময়: ০৯:০০ am থেকে ০৫:০০ pm (সোমবার - শুক্রবার)",
    "question": [
      "বাংলাদেশ হাইকমিশন, আবুজা, নাইজেরিয়ার অফিস সময় কতটা?",
      "বাংলাদেশ হাইকমিশন, আবুজা, নাইজেরিয়া কোন দিনগুলোতে খোলা থাকে?"
    ]
  },
  {
    "annotation_id": "2199400e-ee73-4443-9374-b777c50ee801",
    "text": "বাংলাদেশ ও নাইজেরিয়া ১৯৭০ দশকের মাঝামাঝি সময়ে কূটনৈতিক সম্পর্ক স্থাপন করে। নাইজেরিয়া ১৯৭৪ সালের ২৭ ফেব্রুয়ারি বাংলাদেশকে স্বীকৃতি প্রদান করে।",
    "question": [
      "বাংলাদেশ ও নাইজেরিয়া কবে কূটনৈতিক সম্পর্ক স্থাপন করে?",
      "নাইজেরিয়া বাংলাদেশকে কবে স্বীকৃতি প্রদান করে?"
    ]
  }
]
```

Save this data in a file (e.g., `custom_data.json`) in your project directory.

## Modifying the Script
To evaluate your custom data, you need to update the script to point to your JSON file and specify the embedding models you want to evaluate. Below are the detailed changes required in the script.

### Original Script Snippet (Relevant Section)
The original script specifies the JSON path and embedding models in the `__main__` block:

```python
if __name__ == "__main__":
    # Define paths
    json_path = "data/json___unvalidated__annotation_data_d5f877c8-963e-4a96-925d-6e2c28417895.json"

    # Initialize embedding models
    embedding_models = {
        "base": CustomEmbeddings("l3cube-pune/bengali-sentence-similarity-sbert"),
        "d3_version": CustomEmbeddings("experiments/BanInfRet/d3_baninfret"),
    }

    # Evaluate and plot
    evaluate_multiple_embeddings(json_path, embedding_models)
```

### Changes to Make
1. **Update the JSON Path**:
   Replace the `json_path` with the path to your custom JSON file. For example, if your file is named `custom_data.json` and located in the same directory as the script, update the path as follows:

   ```python
   json_path = "custom_data.json"
   ```

2. **Specify Embedding Models**:
   The script evaluates two embedding models by default: a baseline model and our fine-tuned `d3_version` model. You can keep these models, choose different ones supported by `SentenceTransformer`, or evaluate a single model. For example:
   - To use the baseline model (`l3cube-pune/bengali-sentence-similarity-sbert`) and our fine-tuned `d3_version` model:
     ```python
     embedding_models = {
         "base": CustomEmbeddings("l3cube-pune/bengali-sentence-similarity-sbert"),
         "d3_version": CustomEmbeddings("experiments/BanInfRet/d3_baninfret"),
     }
     ```
   - To evaluate only the `d3_version` model:
     ```python
     embedding_models = {
         "d3_version": CustomEmbeddings("experiments/BanInfRet/d3_baninfret"),
     }
     ```
   Ensure the model paths are valid `SentenceTransformer` models available locally or on Hugging Face. The `d3_version` model, developed by our team, is optimized for Bengali passage retrieval and should be accessible in your local `experiments/BanInfRet/d3_baninfret` directory.

3. **Optional: Customize Output Files**:
   The script saves a Precision-Recall plot (`pr_curve.png`) and metrics (`metrics.json`) by default. To save these files with custom names or in a specific directory, modify the `evaluate_multiple_embeddings` call to include custom `plot_file` and `metrics_file` parameters:
   ```python
   evaluate_multiple_embeddings(
       json_path,
       embedding_models,
       plot_file="output/custom_pr_curve.png",
       metrics_file="output/custom_metrics.json"
   )
   ```
   Ensure the `output/` directory exists or adjust the path accordingly.

4. **Optional: Adjust Thresholds and Top-k Values**:
   The script uses default similarity thresholds (`[0.5, 0.6, 0.7, 0.8, 0.9]`) and top-k values (`[1, 3, 5]`). To customize these, pass them explicitly in the `evaluate_multiple_embeddings` call:
   ```python
   evaluate_multiple_embeddings(
       json_path,
       embedding_models,
       top_k_list=[1, 5, 10],
       thresholds=[0.4, 0.6, 0.8],
       plot_file="output/custom_pr_curve.png",
       metrics_file="output/custom_metrics.json"
   )
   ```

### Updated Script Snippet
Here’s how the updated `__main__` block might look with the changes:

```python
if __name__ == "__main__":
    # Define paths
    json_path = "custom_data.json"

    # Initialize embedding models
    embedding_models = {
        "base": CustomEmbeddings("l3cube-pune/bengali-sentence-similarity-sbert"),
        "d3_version": CustomEmbeddings("experiments/BanInfRet/d3_baninfret"),
    }

    # Ensure output directory exists
    import os
    os.makedirs("output", exist_ok=True)

    # Evaluate and plot
    evaluate_multiple_embeddings(
        json_path,
        embedding_models,
        top_k_list=[1, 5, 10],
        thresholds=[0.4, 0.6, 0.8],
        plot_file="output/custom_pr_curve.png",
        metrics_file="output/custom_metrics.json"
    )
```

## Running the Script
1. Save your JSON data in `custom_data.json`.
2. Update the script with the changes above, ensuring the full script (including imports, `CustomEmbeddings`, `create_vector_store_from_json`, and `evaluate_multiple_embeddings`) is intact.
3. Run the script:
   ```bash
   python script.py
   ```
4. Check the output:
   - A Precision-Recall plot will be saved at `output/custom_pr_curve.png`.
   - Metrics (Recall@k, Precision@k, MRR@k, F1@k, and average top-1 similarity) will be saved in `output/custom_metrics.json`.
   - Console output will display progress and metrics for each model and threshold.

## Troubleshooting
- **File Not Found Error**: Ensure `custom_data.json` is in the correct directory or provide the full path (e.g., `/path/to/custom_data.json`).
- **Model Not Found**: Verify that the specified `SentenceTransformer` model paths are valid and accessible. For the `d3_version` model, ensure the `experiments/BanInfRet/d3_baninfret` directory exists and contains the fine-tuned model.
- **JSON Format Errors**: Ensure the JSON file is well-formed and contains the required keys (`annotation_id`, `text`, `question`). Use a JSON validator if needed.
- **Memory Issues**: If processing a large dataset, reduce the `chunk_size` in `RecursiveCharacterTextSplitter` (e.g., from 1500 to 1000) to lower memory usage.

## Expected Output
The script will:
- Create temporary vector stores for each embedding model.
- Evaluate retrieval performance for each question in the dataset.
- Print metrics (Recall@k, Precision@k, MRR@k, F1@k, Avg Top-1 Similarity) for each model and threshold.
- Save a Precision-Recall plot and a JSON file with detailed metrics.

By following these instructions, you can evaluate your custom dataset with the provided script and compare the performance of different embedding models, including our fine-tuned `d3_version` model, for passage retrieval.