from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import dotenv, os, json
import matplotlib.pyplot as plt
from collections import Counter
dotenv.load_dotenv()

es = Elasticsearch(
    os.getenv("ES_URL"),            
    api_key=os.getenv("ES_API_KEY"),   
    verify_certs=False                  
)
#########透過chunk資料生成法條array#########
def generate_lawsArray_by_chunks(input_file_path):
    df = pd.read_csv(input_file_path)  # Read CSV instead of Excel
    laws = {}
    cnt = [0] * len(df["大塊"])
    keywords = ["第184條", "第185條", "第187條", "第188條", "第191條", "第193條", "第195條", "第213條", "第216條", "第217條"]
    for index, content in enumerate(df["大塊"]):
        id = int(df["大塊第幾筆"][index])
        cnt[id] += 1
        if cnt[id] == 2:
            ls = [0] * len(keywords)
            for i, keyword in enumerate(keywords):
                if keyword in content:
                    ls[i] = 1
            laws[id] = ls
    return laws

#########透過 https://docs.google.com/spreadsheets/d/1w1yoYwDd6Mbv8_3CXgzwmzRX0Nzke8cbCqf2hcDI47w/edit?gid=1273457111#gid=1273457111 生成法條array#########
def generate_lawsArray_by_website(output_file_path):
    df = pd.read_csv(output_file_path)[5:-1]
    laws = []
    keywords = ["第184條", "第185條", "第187條", "第188條", "第191條", "第193條", "第195條", "第213條", "第216條", "第217條"]
    for lawsuit in df["claude-3-5-sonnet-20240620-oneshot.2"]:
        ls = [0] * len(keywords)
        for i, keyword in enumerate(keywords):
            if keyword in lawsuit:
                ls[i] = 1
        laws.append(ls)
    return laws

#########生成要放進elasticsearch的record#########
def generate_records_by_chunks(input_file_path, laws):
    df = pd.read_csv(input_file_path)  # Read CSV instead of Excel
    cnt = [0] * len(df["大塊"])
    chunks = []
    length = len(df["小塊"])
    for index in range(length):
        id = int(df["大塊第幾筆"][index])
        chunks.append({
            "doc_id": id,
            "my_text": df["小塊"][index],
            "my_vector": json.loads(df["小塊EMBEDDING"][index]),  # Deserialize the JSON string back to a list
            "doc_law_array": laws[id],
            "chunk_seq": cnt[id],
        })
        cnt[id] += 1
    return chunks

#########定義elastic search的attribute#########
def create_index(index_name):
    es.indices.create(
        index=index_name,
        mappings={
            "properties": {
                "doc_id": {
                    "type": "integer"
                },
                "my_vector": {
                    "type": "dense_vector",
                    "dims": 1792,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "bbq_hnsw"
                    }
                },
                "doc_law_array":{
                    "type": "dense_vector",
                    "dims": 10,
                },
                "my_text": {
                    "type": "text"
                },
                "chunk_seq": {
                    "type": "integer"
                },
            }
        },
    )

#########將record放進elasticsearch#########
def insert_records(records, index_name):
    for record in records:
            custom_id = f"{record["doc_id"]}_{record["chunk_seq"]}"
            es.index(index=index_name,id=custom_id, body=record)

#########刪除舊的index#########
def delete_old_index(index_name):
    es.indices.delete(index=index_name)

#########用knn search找出最相似的文本#########
def knn_search(index_name, file_path):
    model_name = "TencentBAC/Conan-embedding-v1"  
    model = SentenceTransformer(model_name)
    df = pd.read_csv(file_path)
    laws = generate_lawsArray_by_website(file_path)# 測試起訴狀的法條
    sentences = []
    f1_scores = []
    sentences = df["claude-3-5-sonnet-20240620-oneshot.1"][5:-1].tolist() #模擬律師輸入
    embeddings = model.encode(sentences)
    for index, embedding in enumerate(embeddings):
        resp = es.search(
            index=index_name,
            knn={
                "field": "my_vector",
                "query_vector": embedding,
                "k": 10,
            },
            _source=False,
            fields=[
                "doc_id",
                "doc_law_array",
                "my_text"
            ],
        )
        ######y_predict 為10條knn最相近法條相加後 > 5 的話為1，否則為0######
        y_predict = [0] * 10 
        for content in resp["hits"]["hits"]:
            # print(content["fields"]["doc_law_array"])
            y_predict = [x + y for x, y in zip(y_predict, content["fields"]["doc_law_array"])]
        y_predict = [1 if x >= 5 else 0 for x in y_predict]
        y_true = laws[index]
        f1_scores.append(f1_score(y_true, y_predict))
        print(resp)
        # print(y_predict, y_true, f1_score(y_true, y_predict))
    score_counts = Counter(f1_scores)
    # Sort scores for a cleaner bar chart
    sorted_scores = sorted(score_counts.keys())
    counts = [score_counts[score] for score in sorted_scores]

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_scores, counts, color='skyblue', width=0.02) 

    # Annotate bars with counts
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,  
                str(count), ha='center', fontsize=10)

    # Add titles and labels
    plt.title('Distribution of F1 Scores Across Cases', fontsize=16)
    plt.xlabel('F1 Score', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)

    # Display the chart
    plt.xticks(sorted_scores, rotation=45)  # Rotate X-axis labels for better readability
    plt.ylim(0, max(counts) + 1)  # Add some padding above the highest bar
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better clarity
    plt.tight_layout()

    # Show and save the chart
    plt.show()
    plt.savefig('f1_scores_chart.png')
input_file_path = 'EmbedForCSV.csv'
output_file_path = '起訴狀案例測試 - 5. 判決書500筆(實469).csv'
output_file_path2 = '起訴狀案例測試 - 6. 判決書500筆(實461).csv'
small_chunks_index_name = "lawsuit_small_chunks_bbq"
small_slow_chunks_index_name = "lawsuit_small_slow_chunks"
laws = generate_lawsArray_by_chunks(input_file_path)
records = generate_records_by_chunks(input_file_path, laws)
print(len(records))
# delete_old_index(small_chunks_index_name)
# create_index(small_slow_chunks_index_name)
# insert_records(records, small_slow_chunks_index_name)
# knn_search(small_slow_chunks_index_name, output_file_path2)
