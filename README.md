## PaperHelper: Knowledge-Based LLM QA Paper Reading Assistant with Reliable References

### [1] Introduction

Thanks to the Great [Ms. Freax](https://github.com/H-Freax) for designing the [Athenas-Oracle project](https://github.com/H-Freax/Athenas-Oracle). 

Based on it, we have made improvements and designed the Paper Helper for Machine Learning Scientists. With the effects of RAG Fusion and RAFT (RAG Finetune, fine-tuned using GPT-4-1106-Preview API on the 52,000 [MLArxivPapers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) and [ArxivQA](https://huggingface.co/datasets/arxiv-community/arxiv_dataset) dataset as the backend), it can effectively reduce hallucinations and enhance retrieval relevance. We have implemented an end-to-end application of parallel generating, providing useful information to paper readers based on references ranked by relevance. We also incorporated structural relationships to represent the extracted information. 

In short, everything is designed to enable a machine learning researcher to read papers more efficiently and provide the most reliable references based on paper citations!

### [2] Implementation Details

### Overview
The assistant utilizes three tools: search, gather evidence, and answer questions. These tools enable it to find and parse relevant full-text research papers, identify specific sections in the paper that help answer the question, summarize those sections with the context of the question (called evidence), and then generate an answer based on the evidence. It is an agent so that the LLMs orchestrating the tools can adjust the input to paper searches, gather evidence with different phrases, and assess if an answer is complete. 
<div align=center>
	<img src="https://github.com/JerryYin777/PaperHelper/assets/88324880/a66103ea-58ed-4daa-b56e-8f5615f816c5"/>
</div>

### Basic RAG
The basic RAG simply splits the search prompt into simple words in a crude manner, and may produce certain spelling illusions without truly understanding the user's intent.
![Basic RAG](https://github.com/JerryYin777/PaperHelper/assets/88324880/3a39564d-3cbf-49c5-b5ae-7f0888d40039)

### RAG Fusion with RAFT
Our system also has integrated the [RAFT](https://arxiv.org/pdf/2403.10131) method. This approach enhances the capability of LLMs in specific RAG tasks by leveraging the core idea that if LLMs can "learn" documents in advance, it can improve RAG's performance. 

We finetuned the OpenAI API using 52,000 domain-specific papers from the field of machine learning to augment the knowledge of PaperHelper within the machine learning domain, thereby assisting machine learning scientists in reading papers more efficiently and accurately.

<div align=center>
	<img src="https://github.com/JerryYin777/PaperHelper/assets/88324880/85816fc0-487a-4460-ad8c-a82c9d8ff323"/>
</div>

### Extract Relevance
With the implementation of RAFT, we can extract the reference section at the end of articles more efficiently. First, we use RAG to traverse all the references in the article. Then, based on the knowledge from the LLMs, we refine the information using the top-k algorithm to identify the literature most relevant to the article.

<div align=center>
	<img src="https://github.com/JerryYin777/PaperHelper/assets/88324880/c5b232cb-b236-4d4e-8b21-860749b64ca1"/>
</div>

We can find that through the RAFT method, the model integrates cutting-edge knowledge, enabling readers to further explore academic papers based on current information rather than providing outdated and misleading content.

<div align=center>
	<img src="https://github.com/JerryYin777/PaperHelper/assets/88324880/e535614f-f07e-4107-aeed-01e63dae66fb"/>
</div>

### [3] Usage
Use the following command step by step:
1. **Clone the Repository**
```bash
git clone https://github.com/JerryYin777/PaperHelper.git
```
2. **Install Dependencies**
```bash
cd PaperHelper
pip install -r requirements.txt
```
3. **Set OpenAI API Key**
```bash
cd .streamlit
touch secrets.toml #input your OPENAI_API_KEY = "sk-yourapikeyhere" here
```
4. **Start PaperHelper**
```bash
streamlit run app.py
```
**Note:** 
1. Set `allow_dangerous_deserialization: bool = True` first, where you can find in `faiss.py`.

<div align=center>
	<img src="https://github.com/JerryYin777/PaperHelper/assets/88324880/2669ad40-e3c5-4a48-b393-4ffdb4709231"/>
</div>

2. You may also embed your pdf first in the application (click the button), or you may raise error `Exceptation: Directory index does not exist.`




