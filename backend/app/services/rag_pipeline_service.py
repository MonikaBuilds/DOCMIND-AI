from dataclasses import dataclass

from app.domain.chat import ChatMessage, CitedAnswer
from app.domain.chunks import DocumentChunk
from app.domain.documents import Document
from app.domain.retrieval import RetrievedChunk
from app.services.answer_service import AnswerService
from app.services.chunking_service import ChunkingConfig, ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.metadata_service import MetadataService
from app.services.parsing_service import ParsingService
from app.services.preprocessing_service import PreprocessingService
from app.services.prompt_service import PromptService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalConfig, RetrievalService
from app.services.summary_service import SummaryService
from app.services.vector_service import VectorService


@dataclass(frozen=True)
class IndexedDocument:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    chunks: list[DocumentChunk]
    ai_summary: str | None = None


class RAGPipelineService:
    """End-to-end RAG orchestration for indexing documents and answering questions."""

    def __init__(
        self,
        parsing_service: ParsingService,
        preprocessing_service: PreprocessingService,
        chunking_service: ChunkingService,
        metadata_service: MetadataService,
        embedding_service: EmbeddingService,
        vector_service: VectorService,
        retrieval_service: RetrievalService,
        reranking_service: RerankingService,
        answer_service: AnswerService,
        summary_service: SummaryService | None = None,
    ) -> None:
        self.parsing_service = parsing_service
        self.preprocessing_service = preprocessing_service
        self.chunking_service = chunking_service
        self.metadata_service = metadata_service
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.retrieval_service = retrieval_service
        self.reranking_service = reranking_service
        self.answer_service = answer_service
        self.summary_service = summary_service or SummaryService(self.answer_service.llm_service)

    @classmethod
    def from_providers(cls, embedding_provider, vector_store, llm_provider, chunk_config: ChunkingConfig):
        embedding_service = EmbeddingService(embedding_provider)
        vector_service = VectorService(vector_store)
        retrieval_service = RetrievalService(
            embedding_service,
            vector_service,
            RetrievalConfig(default_top_k=5, max_top_k=20),
        )
        return cls(
            parsing_service=ParsingService(),
            preprocessing_service=PreprocessingService(),
            chunking_service=ChunkingService(chunk_config),
            metadata_service=MetadataService(),
            embedding_service=embedding_service,
            vector_service=vector_service,
            retrieval_service=retrieval_service,
            reranking_service=RerankingService(),
            answer_service=AnswerService(PromptService(), LLMService(llm_provider)),
        )

    def index_document(self, document: Document) -> IndexedDocument:
        parsed_document = self.parsing_service.parse_document(document)
        cleaned_document = self.preprocessing_service.clean_document(parsed_document)
        chunks = self.chunking_service.chunk_document(cleaned_document)
        chunks = self.metadata_service.propagate_headings(chunks)
        embeddings = self.embedding_service.embed_chunks(chunks)
        self.vector_service.add_chunks(chunks, embeddings)
        
        ai_summary = self.summary_service.generate_document_summary(cleaned_document)

        return IndexedDocument(
            document_id=document.document_id,
            filename=document.filename,
            page_count=cleaned_document.metadata.page_count,
            chunk_count=len(chunks),
            chunks=chunks,
            ai_summary=ai_summary,
        )

    def answer_question(
        self,
        question: str,
        conversation_history: list[ChatMessage] | None = None,
        top_k: int = 5,
    ) -> tuple[CitedAnswer, list[RetrievedChunk]]:
        retrieved = self.retrieval_service.semantic_search(question, top_k=top_k)
        reranked = self.reranking_service.rerank(question, retrieved, top_k=top_k)
        answer = self.answer_service.generate_answer(
            question=question,
            retrieved_chunks=reranked,
            conversation_history=conversation_history
        )
        return answer, reranked
