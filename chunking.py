import re, json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

def embed_texts(texts):
    embed = HuggingFaceEmbeddings(
        model_name="TencentBAC/Conan-embedding-v1",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return [embed.embed_query(text) for text in texts]

try:
    # Initialize the Hugging Face embedding model
    embed = HuggingFaceEmbeddings(
        model_name="TencentBAC/Conan-embedding-v1",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    
    # Read the input file
    df = pd.read_csv('起訴狀案例測試 - 5. 判決書500筆(實469).csv')
    cases = df["claude-3-5-sonnet-20240620-oneshot.2"][5:-2].tolist()
    # with open('example_cases copy.txt', 'r', encoding='utf-8') as file2:
    #     content2 = file2.read()
    
    # # Split cases
    # cases = [case.strip() for case in content2.split('"') if case.strip()]
    # Lists to store data for both chunking methods

    large_chunks = []  # For original (大塊) method
    small_chunks = []  # For new (小塊) method
    
    # Process each case
    for case_id, case in enumerate(cases):
        print(case_id)
        print(case[0:100])
        # Original chunking method (大塊)
        match = re.search(r'一、(.*?)二、(.*)', case, re.S)
        if match:
            fact_text = match.group(1).strip()
            remaining_text = match.group(2).strip()
            comp_match = re.search(r'\（\s*一\s*\）', remaining_text)
            
            if comp_match:
                legal_text = remaining_text[:comp_match.start()].strip()
                compensation_text = remaining_text[comp_match.start():].strip()
            else:
                legal_text = remaining_text
                compensation_text = ""
            
            # Generate embeddings for original chunks
            if fact_text:
                fact_embedding = embed.embed_query(fact_text)
                large_chunks.append([case_id, fact_text, fact_embedding])
            else:
                large_chunks.append([case_id, "", []])
            
            if legal_text:
                legal_embedding = embed.embed_query(legal_text)
                large_chunks.append([case_id, legal_text, legal_embedding])
            else:
                large_chunks.append([case_id, "", []])
                
            if compensation_text:
                comp_embedding = embed.embed_query(compensation_text)
                large_chunks.append([case_id, compensation_text, comp_embedding])
            else:
                large_chunks.append([case_id, "", []])
        
        # New sentence-based chunking method (小塊)
        single_sentences_list = re.split(r'[，。]', case)
        sentences = [{'sentence': x.strip(), 'index': i} for i, x in enumerate(single_sentences_list) if x.strip()]
        
        # Generate embeddings for sentences
        embeddings = embed_texts([x['sentence'] for x in sentences])
        for i, sentence in enumerate(sentences):
            sentence['embedding'] = embeddings[i]
        
        # Calculate cosine distances
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['embedding']
            embedding_next = sentences[i + 1]['embedding']
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            distances.append(similarity)
            sentences[i]['distance_to_next'] = similarity
        
        # Calculate threshold and create chunks
        if distances:
            sorted_distances = sorted(distances)
            percentage = 90
            cutoff_index = int(len(sorted_distances) * (100 - percentage) / 100)
            threshold = sorted_distances[cutoff_index]
            
            # Determine breakpoints
            indices_above_thresh = [i for i, x in enumerate(distances) if x < threshold]
            
            # Create chunks
            start_index = 0
            chunks = []
            
            # Process breakpoints
            for index in indices_above_thresh:
                group = sentences[start_index:index + 1]
                combined_text = '。'.join([d['sentence'] for d in group]) + '。'
                chunks.append(combined_text)
                start_index = index + 1
            
            # Add final chunk if needed
            if start_index < len(sentences):
                combined_text = '。'.join([d['sentence'] for d in sentences[start_index:]]) + '。'
                chunks.append(combined_text)
            
            # Generate embeddings for chunks
            chunk_embeddings = embed_texts(chunks)
            # Add small chunks to list
            for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
                small_chunks.append([case_id, chunk, chunk_embedding])

        else:
            # Add empty chunk if no valid sentences found
            small_chunks.append([case_id, "", []])
    
    # Create DataFrames for both chunking methods
    df_large = pd.DataFrame(large_chunks, columns=['大塊第幾筆', '大塊', '大塊EMBEDDING'])
    df_small = pd.DataFrame(small_chunks, columns=['小塊第幾筆', '小塊', '小塊EMBEDDING'])
    # Initialize an empty DataFrame with all required columns
    all_indexes = sorted(list(set(df_large['大塊第幾筆'].unique()) | set(df_small['小塊第幾筆'].unique())))
    df_combined = pd.DataFrame(index=range(len(all_indexes)))
    
    # Fill in data from both DataFrames
    for idx in all_indexes:
        large_rows = df_large[df_large['大塊第幾筆'] == idx]
        small_rows = df_small[df_small['小塊第幾筆'] == idx]
        
        # Create rows for this index
        max_rows = max(len(large_rows), len(small_rows))
        for i in range(max_rows):
            new_row = {
                '大塊第幾筆': int(idx),
                '大塊': large_rows.iloc[i]['大塊'] if i < len(large_rows) else "",
                '大塊EMBEDDING': large_rows.iloc[i]['大塊EMBEDDING'] if i < len(large_rows) else [],
                '小塊第幾筆': int(idx),
                '小塊': small_rows.iloc[i]['小塊'] if i < len(small_rows) else "",
                '小塊EMBEDDING': small_rows.iloc[i]['小塊EMBEDDING'] if i < len(small_rows) else []
            }
            df_combined = pd.concat([df_combined, pd.DataFrame([new_row])], ignore_index=True)
    
    # Remove the initial empty rows created during DataFrame initialization
    df_combined = df_combined.dropna(how='all')
    # Serialize embeddings (lists) as JSON strings for CSV compatibility
    df_combined['大塊EMBEDDING'] = df_combined['大塊EMBEDDING'].apply(json.dumps)
    df_combined['小塊EMBEDDING'] = df_combined['小塊EMBEDDING'].apply(json.dumps)

    # Save to CSV
    df_combined.to_csv('embedForcsv.csv', index=False, encoding='utf-8-sig')
    print("CSV file has been created successfully!")

except FileNotFoundError:
    print("Error: csv file not found")
except Exception as e:
    print(f"An error occurred: {str(e)}")

"""import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

def embed_texts(texts):
    embed = HuggingFaceEmbeddings(
        model_name="TencentBAC/Conan-embedding-v1",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return [embed.embed_query(text) for text in texts]

try:
    # Initialize the Hugging Face embedding model
    embed = HuggingFaceEmbeddings(
        model_name="TencentBAC/Conan-embedding-v1",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Read the input file
    with open('example_cases.txt', 'r', encoding='utf-8') as file2:
        content2 = file2.read()
    
    # Split cases
    cases = [case.strip() for case in content2.split('"') if case.strip()]
    
    # Lists to store data for both chunking methods
    large_chunks = []  # For original (大塊) method
    small_chunks = []  # For new (小塊) method
    
    # Process each case
    for case_id, case in enumerate(cases):
        # Original chunking method (大塊)
        match = re.search(r'一、(.*?)二、(.*)', case, re.S)
        if match:
            fact_text = match.group(1).strip()
            remaining_text = match.group(2).strip()
            comp_match = re.search(r'\（\s*一\s*\）', remaining_text)
            
            if comp_match:
                legal_text = remaining_text[:comp_match.start()].strip()
                compensation_text = remaining_text[comp_match.start():].strip()
            else:
                legal_text = remaining_text
                compensation_text = ""
            
            # Generate embeddings for original chunks
            if fact_text:
                fact_embedding = embed.embed_query(fact_text)
                large_chunks.append([case_id, fact_text, fact_embedding])
            
            if legal_text:
                legal_embedding = embed.embed_query(legal_text)
                large_chunks.append([case_id, legal_text, legal_embedding])
            
            if compensation_text:
                comp_embedding = embed.embed_query(compensation_text)
                large_chunks.append([case_id, compensation_text, comp_embedding])
        
        # New sentence-based chunking method (小塊)
        single_sentences_list = re.split(r'[，。]', case)
        sentences = [{'sentence': x.strip(), 'index': i} for i, x in enumerate(single_sentences_list) if x.strip()]
        
        # Generate embeddings for sentences
        embeddings = embed_texts([x['sentence'] for x in sentences])
        for i, sentence in enumerate(sentences):
            sentence['embedding'] = embeddings[i]
        
        # Calculate cosine distances
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['embedding']
            embedding_next = sentences[i + 1]['embedding']
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            distances.append(similarity)
            sentences[i]['distance_to_next'] = similarity
        
        # Calculate threshold and create chunks
        if distances:
            sorted_distances = sorted(distances)
            percentage = 90
            cutoff_index = int(len(sorted_distances) * (100 - percentage) / 100)
            threshold = sorted_distances[cutoff_index]
            
            # Determine breakpoints
            indices_above_thresh = [i for i, x in enumerate(distances) if x < threshold]
            
            # Create chunks
            start_index = 0
            chunks = []
            
            # Process breakpoints
            for index in indices_above_thresh:
                group = sentences[start_index:index + 1]
                combined_text = '。'.join([d['sentence'] for d in group]) + '。'
                chunks.append(combined_text)
                start_index = index + 1
            
            # Add final chunk if needed
            if start_index < len(sentences):
                combined_text = '。'.join([d['sentence'] for d in sentences[start_index:]]) + '。'
                chunks.append(combined_text)
            
            # Generate embeddings for chunks
            chunk_embeddings = embed_texts(chunks)
            
            # Add small chunks to list
            for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
                small_chunks.append([case_id, chunk, chunk_embedding])
    
    # Create DataFrames for both chunking methods
    df_large = pd.DataFrame(large_chunks, columns=['大塊第幾筆', '大塊', '大塊EMBEDDING'])
    df_small = pd.DataFrame(small_chunks, columns=['小塊第幾筆', '小塊', '小塊EMBEDDING'])
    
    # Combine the DataFrames
    # Use merge to combine based on case IDs, or create a full cross product if needed
    df_combined = pd.merge(
        df_large, 
        df_small, 
        left_on='大塊第幾筆',
        right_on='小塊第幾筆',
        how='outer'
    )
    
    # Save to Excel
    df_combined.to_excel('EmbedForExcel.xlsx', index=False)
    print("Excel file has been created successfully!")

except FileNotFoundError:
    print("Error: example_cases.txt file not found")
except Exception as e:
    print(f"An error occurred: {str(e)}")"""