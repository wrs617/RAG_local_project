from gitdb.fun import chunk_size
from rapidfuzz.distance.Levenshtein_py import similarity
from torch.fx.experimental.meta_tracer import embedding_override

md5_path = "./md5.txt"
#chroma
collection_name = 'rag'
persist_directory = './chroma_db'
#spliter
chunk_size = 1000
chunk_overlap = 100
separators=["\n\n","\n",".","!","?","。","!","？"," "]
max_split_char_number = 1000   #文本分割的阈值

similarity_threshold = 2  #检索返回匹配额文档数量如果改为1就是最精准的匹配结果

embedding_model_name = "text-embedding-v4"
DASHSCOPE_API_KEY = "sk-your api key"


#在赋值语句中 x=“123”，   此时x是元组
#embedding_model_name = "text-embedding-v4" 不能加逗号