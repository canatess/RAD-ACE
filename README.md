# 🏥 RAD-ACE: MLLMs AS RADIOLOGY ASSISTANTS

### 📌 CMP719 Computer Vision Project  
#### ✍️ Authors: **Can Ali ATEŞ, Emre ÇOBAN, Abdullah Enes ERGÜN**  
⚠️ **Important Note**
> The training stages of the models were carried out using Google Colab Pro+ due to GPU issues. Due to a mismatch in notebook metadata between Colab and Github, direct viewing via Github is not possible (Invalid Notebook), so you will need to download the files.

## 🧐 Overview  
**RAD-ACE** is a cutting-edge research project designed to **fine-tune vision-language large language models (VLLMs)** for **structured medical report generation**. By integrating advanced **computer vision** with **language models**, it aims to produce **accurate, coherent, and context-aware** medical analyses.  

This project focuses on enhancing the **logical structuring** of AI-generated reports, ensuring **interpretability, clinical reliability, and consistency** in diagnostic documentation.  

## 🧠 Models Used  
- 🔹 **Qwen 2.5 VL-3B**: Lightweight VLM used for efficient inference and baseline evaluation on vision-language tasks.  
- 🔹 **Qwen 2.5 VL-7B**: Larger variant for improved multimodal reasoning and image-text alignment. 
- 🔹 **LLaMA 3.2 Vision - 11B**: High-capacity model used for advanced multimodal understanding, including visual question answering and report generation. 

## 📂 Dataset Sources  
- 🔗 **Dataset 1**: [[PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision)]  
- 🔗 **Dataset 2**: [[VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)]  

## 📄 Research Paper  
For detailed insights into our methodology and findings, refer to our research paper:  
📌 **[RAD-ACE: MLLMs AS RADIOLOGY ASSISTANTS]** – [Paper Link](https://github.com/canatess/RAD-ACE/blob/2d933bbc689de53925940de3086b3e888759db0f/Final%20Report.pdf)

## ⚙️ Installation & Usage  
1️⃣ **Clone the repository:**  
   ```bash
   git clone https://github.com/canatess/RAD-ACE.git
