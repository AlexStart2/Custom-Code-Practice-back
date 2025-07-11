from typing import List
from unstructured.partition.auto import partition
from fastapi import UploadFile
from unstructured.partition.auto import partition
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import subprocess, json, os

import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch




def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]





async def process_documents_in_memory(
    files: List[UploadFile],
    clip_model: str = 'intfloat/multilingual-e5-large-instruct',
    st_model: SentenceTransformer = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
) -> List[dict]:
    """
    For each UploadFile in `files`:
      1) Read bytes, write to a temporary buffer
      2) Partition (extract) text
      3) Chunk the text
      4) Embed each chunk via CLIP (Ollama) & ST
      5) Return a list of dicts with 'source', 'chunk', and both embeddings
    """
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    results = []

    for f in files:
        # 1) Read into bytes buffer
        contents = await f.read()  # in your async endpoint
        # 2) Partition: unstructured supports buffer as well
        elements = partition(filename=f.filename, file=contents)
        text_segments = [el.text for el in elements if getattr(el, 'text', None)]
        full_text = "\n".join(text_segments)

        # 3) Chunk
        chunks = splitter.split_text(full_text)

        # 4) Embed & collect
        for chunk in chunks:
            # CLIP via subprocess
            proc = subprocess.run(
                ['ollama', 'embed', clip_model, '--stdin', '--json'],
                input=chunk.encode('utf-8'),
                capture_output=True,
                check=True
            )
            clip_emb = json.loads(proc.stdout)['embedding']

            # Sentence Transformer
            st_emb = st_model.encode(chunk).tolist()

            results.append({
                'source': f.filename,
                'chunk': chunk,
                'embeddings': {
                    'clip': clip_emb,
                    'sentence': st_emb
                }
            })
    return results



# import torch.nn.functional as F

# from torch import Tensor
# from transformers import AutoTokenizer, AutoModel


# def average_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# def get_detailed_instruct(task_description: str, query: str) -> str:
#     return f'Instruct: {task_description}\nQuery: {query}'

# # Each query must come with a one-sentence instruction that describes the task
# task = 'Given a web search query, retrieve relevant passages that answer the query'
# queries = [
#     get_detailed_instruct(task, 'how much protein should a female eat'),
#     get_detailed_instruct(task, '南瓜的家常做法')
# ]
# # No need to add instruction for retrieval documents
# documents = [
#     "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#     "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
# ]
# input_texts = queries + documents

# tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
# model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')

# # Tokenize the input texts
# batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

# outputs = model(**batch_dict)
# embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# # normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:2] @ embeddings[2:].T) * 100
# print(scores.tolist())