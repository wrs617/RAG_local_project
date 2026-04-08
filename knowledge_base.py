import hashlib
import os
from datetime import datetime
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config


def check_md5(md5_str: str):
    # 检查传入的md5字符串是否已经被处理过了,如果return false MD5未处理过
    if not os.path.exists(config.md5_path):
        # if进入表示文件不存在，没有处理过文件
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()  # 处理字符串前后空格和回车
            if line == md5_str:
                return True  # 已经处理过
        return False


def save_md5(md5_str):
    # 将传入的md5字符串，记录保存到文件内
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    # 将传入的字符串转换为md5字符串
    # 将字符串转换为bytes数组
    str_bytes = input_str.encode(encoding=encoding)
    # 创建MD5对象
    md5_obj = hashlib.md5()
    # 得到对象
    md5_obj.update(str_bytes)
    # 更新内容
    md5_hex = md5_obj.hexdigest()
    return md5_hex


class KnowledgeBaseService(object):
    def __init__(self):
        # 如果文件夹不存在则创建，如国存在就跳过
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,  # 数据库的表明
            embedding_function=DashScopeEmbeddings(
                model="text-embedding-v4",
                dashscope_api_key="sk-33b1b97f1d5e42329aae6d8a93d2c9d3"
            ),
            persist_directory=config.persist_directory,  # 数据库存储文件夹
        )  # 向量存储的实例 Chroma向量库对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,  # 分割后文本最大长度
            chunk_overlap=config.chunk_overlap,  # 连续文本
            separators=config.separators,  # 自然段落划分的符号
            length_function=len,  # 使用python自带的len函数做长度统计的依据
        )  # 文本分割器的对象

    def upload_by_str(self, data, filename):
        # --- 修改开始：将去重逻辑移到最顶端，无论长短都检查 ---
        # 1. 先计算当前传入内容的 MD5
        md5_hex = get_string_md5(data)
        # 2. 检查该 MD5 是否已存在
        if check_md5(md5_hex):
            return "[跳过]内容已经存在在知识库中"
        # --- 修改结束 ---

        # --- 原有的分割逻辑保持不变 (去掉了里面的去重判断) ---
        # 注意：这里不再需要 if check_md5 了，因为上面已经检查过了
        if len(data) > config.max_split_char_number:
            knowledge_chunks = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            "creat_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "小王"
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        # --- 修改开始：将记录 MD5 放到最后 ---
        # 只有入库成功后，才记录这个 MD5，防止出错
        save_md5(md5_hex)
        # --- 修改结束 ---

        return "[成功]内容已经成功载入到向量库"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰伦", "testfile")
    print(r)