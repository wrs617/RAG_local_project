from langchain_chroma import Chroma
import config_data as config



class VectoryService(object):
    def __init__(self,embedding):

        self.embedding = embedding

        self.vector_store = Chroma(
             collection_name=config.collection_name,
             embedding_function=self.embedding,
             persist_directory=config.persist_directory,

        )
    def get_retriever(self):
        '''返回向量检索器，方便加入链'''
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})

if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectoryService(embedding=DashScopeEmbeddings(model="text-embedding-v4",dashscope_api_key="sk-your api key")).get_retriever()
    res = retriever.invoke("我是成年男性，体重140斤，给我推荐一个上衣")
    print(res)