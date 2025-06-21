import os
import pandas as pd
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI

# === SETUP ===

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === LOAD DATA ===

qa_df = pd.read_csv("question_answers.csv")
qa_df.rename(
    columns={"ID": "image_id", "Question": "question", "Answer": "ground_truth"},
    inplace=True,
)

# Load model outputs
qwen3b_df = pd.read_csv("qwen3b_inference_results.csv")
qwen7b_df = pd.read_csv("qwen7b_inference_results.csv")
llama_df = pd.read_csv("llama11b_inference_results.csv")

# Add model names
qwen3b_df["model"] = "Qwen3B"
qwen7b_df["model"] = "Qwen7B"
llama_df["model"] = "LLaMA11B"

# Combine and align image IDs
all_models_df = pd.concat([qwen3b_df, qwen7b_df, llama_df])
all_models_df["image_id"] = (
    all_models_df["image_name"].str.replace(".jpg", "", regex=False).astype(int)
)

# Merge with Q&A
merged_df = pd.merge(all_models_df, qa_df, on="image_id", how="left")


# === UTILITIES ===


def encode_image(img_path):
    with Image.open(img_path) as img:
        buffered = BytesIO()
        img.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def create_prompt(image_name, question, model_output, ground_truth):
    return f"""
You are a medical expert trained in radiology and LLM evaluation. Your task is to evaluate a model-generated output for a medical visual question answering (VQA) task.

📌 Image filename: {image_name}

🧠 Question:
{question}

✅ Ground Truth Answer:
{ground_truth}

📝 Model Output:
{model_output}

Please evaluate the model's output based on the following five criteria, each on a scale of 1 (poor) to 5 (excellent):

1. Clinical Relevance — Is the reasoning focused on medically relevant findings?
2. Factuality / Hallucination — Are all claims strictly grounded in the image and question? No fabrication?
3. Reasoning Coherence — Is the reasoning step-by-step, logical, and easy to follow?
4. Completeness — Are all key aspects of the question and image addressed?
5. Final Answer Quality — Is the final answer clinically sound, clearly presented, and aligned with reasoning?

⚠️ Return ONLY the evaluation as a **valid, compact JSON object** with integer values for each criterion. Do NOT include explanations, notes, markdown, or extra commentary.

⚠️ Format your entire response **exactly** as:
{{
  "clinical_relevance": 1-5,
  "factuality": 1-5,
  "reasoning_coherence": 1-5,
  "completeness": 1-5,
  "final_answer_quality": 1-5
}}
"""


# === EVALUATION LOOP ===

results = []

for i, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    try:
        img_path = f"photos/{row['image_name']}"
        image_base64 = encode_image(img_path)

        prompt = create_prompt(
            row["image_name"], row["question"], row["model_output"], row["ground_truth"]
        )

        messages = [
            {
                "role": "system",
                "content": "You are a strict radiology model evaluator.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ]

        # Run inference
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0
        )

        raw_output = response.choices[0].message.content.strip()

        # Attempt to parse JSON
        try:
            scores = json.loads(raw_output)
            results.append({**row, **scores, "raw_output": raw_output})
        except json.JSONDecodeError as e:
            print(f"[PARSE ERROR] Row {i} - JSON failed: {e}")
            results.append({**row, "raw_output": raw_output, "error": str(e)})

    except Exception as e:
        print(f"[ERROR] Row {i} ({row['image_name']} | {row['model']}):", e)
        results.append({**row, "error": str(e)})


# === SAVE RESULTS ===

final_df = pd.DataFrame(results)
final_df.to_csv("strict_eval_results_openai.csv", index=False)
print("✅ Finished and saved to strict_eval_results_openai.csv")
