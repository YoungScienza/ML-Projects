from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.document_transformers import (
    LongContextReorder,
)
from typing import List, Any


class LongContextReorderRetriever(BaseRetriever):
    
    base_retriever: BaseRetriever
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        
        docs = self.base_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child(), **kwargs
        )
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs)
        
        return reordered_docs