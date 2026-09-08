import os
import re
import json
import time
import logging
import traceback
from pathlib import Path
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import DoclingDocument
from docling.datamodel.base_models import DocumentStream, InputFormat

from .schemas import Chunk, Document, Output, OutputBasic
from .store import chunk_entities_extraction_prompt, chunk_entities_extraction_prompt_basic
from .tokenizer import Tokenizer
from .utils import Timer, generate_sha256_hash, preprocess_text, get_file_hash
import traceback
from requests.exceptions import ConnectionError, Timeout

from openai import RateLimitError, APIConnectionError, APIStatusError
import time
import logging
import traceback

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    APIConnectionError,
    APIStatusError,
    ConnectionError, 
    Timeout,
    json.JSONDecodeError
)

from dotenv import load_dotenv
load_dotenv()

class DocSegmenter(Tokenizer):
    
    
    HEADING_PATTERN: re.Pattern[str] = re.compile(r"##(.*?)(?=\n)")
    
    def __init__(self, model_name : str = "gpt-4o-mini"):
        
        self._llm  = ChatOpenAI(model='gpt-5.4-mini', temperature=0)
        self._llm2  = ChatOpenAI(model=model_name, temperature=0)
        super().__init__()        

    def extract_content(self, filepath=None, * , device : str = 'cpu', file_bytes = None, file_name = None) -> str:
        
        try:    
            _parser = DocumentConverter()
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options.device = device
            _parser = DocumentConverter(format_options={
                            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                            InputFormat.DOCX: WordFormatOption() # Uses SimplePipeline by default
                        })       
            
            if file_bytes is not None:
                return Document(
                    id=generate_sha256_hash(file_bytes),
                    name=file_name,
                    page_count=1,
                    source=file_bytes,
                )
            
            # if file_bytes is not None:
            #     if not file_name:
            #         raise ValueError('file_name is required when passing file_bytes')
            #     source = DocumentStream(name=file_name, stream=BytesIO(file_bytes))
            #     parsed_doc: DoclingDocument = _parser.convert(source).document
            #     md = parsed_doc.export_to_markdown()
            #     return Document(id=generate_sha256_hash(md), name=file_name, page_count=parsed_doc.num_pages(), source=md)

            fp = Path(filepath)
            if fp.exists():
                
                parsed_doc: DoclingDocument = _parser.convert(filepath).document
                md = parsed_doc.export_to_markdown()
                                
                # return Document(id=generate_sha256_hash(md), name = parsed_doc.name, page_count=parsed_doc.num_pages(), source=md)
                return Document(id=generate_sha256_hash(md), name = fp.name, page_count=parsed_doc.num_pages(), source=md)
                
            else:
                raise FileNotFoundError(f"{fp} path doesn't exist ! Please provide a valid document path to extract.")    
            
        except Exception:
            print(f'Error occured while extracting content : \n{traceback.format_exc()}')

    def _title_based_chunking(self, markdown : str) -> tuple[list, list, list]:
        matches = list(DocSegmenter.HEADING_PATTERN.finditer(markdown))
        
        if not matches:
            return self._fallback_paragraph_chunking(markdown)
        
        text_chunks, text_tokens, text_topics = [],[],[]
        total_length = len(markdown)

        # capture any text BEFORE the first heading match
        preamble = markdown[0:matches[0].start()]
        if preamble.strip():
            split_preamble = preamble.replace("\n\n", "\n").split(" ")
            text_chunks.append(split_preamble)
            text_tokens.append([self.count_tokens(ch) for ch in split_preamble])
            text_topics.append("## Untitled Section")

        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i + 1].start() if i + 1 < len(matches) else total_length

            chunk = markdown[start:end].replace("\n\n", "\n")
            split_chunk = chunk.split(" ")
            
            text_chunks.append(split_chunk)
            text_tokens.append([self.count_tokens(ch) for ch in split_chunk])
            text_topics.append(matches[i].group(0))
            
        return text_chunks, text_tokens, text_topics

    def _fallback_paragraph_chunking(self, markdown):
        paragraphs = [p for p in markdown.split("\n\n") if p.strip()]

        if not paragraphs:
            paragraphs = [markdown]
            
        text_chunks, text_tokens, text_topics = [], [], []
        for para in paragraphs:
            split_chunk = para.split(" ")
            text_chunks.append(split_chunk)
            text_tokens.append([self.count_tokens(ch) for ch in split_chunk])
            text_topics.append("## Untitled Section")

        return text_chunks, text_tokens, text_topics   

    # updated chunking code
    def _adjust_token_size(self, text_chunks, text_tokens, text_topics, max_token=512, overlap_tokens=50):
        """
        look-ahead overlap: the end of each chunk repeats the START of the next
        chunk (heading + first bit of its content). The next chunk itself is
        NOT shortened - it always starts fresh and complete. Only the tail end
        of the PREVIOUS chunk borrows a preview of what comes next.

        Two chunking cases per topic:
        1) Topic fits within budget on its own -> merges with adjacent small
            topics into one chunk.
        2) Topic exceeds budget -> split into multiple chunks on its own.
        """
        budget = max_token - overlap_tokens if overlap_tokens < max_token else max_token

        # Phase 1: build chunks with NO overlap yet. Each raw chunk also records
        # its own "head" (topic heading text if it opens a topic + first words)
        # so the PREVIOUS chunk can borrow a preview of it in phase 2.
        raw = []

        current_parts, current_titles, current_tokens = [], [], 0
        current_head_topic, current_head_words, current_head_wtok = None, None, None

        def flush_merged():
            nonlocal current_parts, current_titles, current_tokens
            nonlocal current_head_topic, current_head_words, current_head_wtok
            if not current_parts:
                return
            raw.append({
                "title": " / ".join(current_titles),
                "body": " ".join(current_parts),
                "head_topic": current_head_topic,
                "head_words": current_head_words,
                "head_wtok": current_head_wtok,
            })
            current_parts, current_titles, current_tokens = [], [], 0
            current_head_topic, current_head_words, current_head_wtok = None, None, None

        def split_big_topic(words, w_token, topic):
            n, i, first = len(words), 0, True
            while i < n:
                j, token_sum = i, 0
                while j < n and token_sum + w_token[j] <= budget:
                    token_sum += w_token[j]
                    j += 1
                if j == i:
                    j = i + 1
                slice_words = words[i:j]
                body = ' '.join(slice_words) if first else " ".join(slice_words)
                raw.append({
                    "title": topic,
                    "body": body,
                    "head_topic": topic if first else None,
                    "head_words": slice_words,
                    "head_wtok": w_token[i:j],
                })
                first = False
                i = j

        for words, w_token, topic in zip(text_chunks, text_tokens, text_topics):
            topic = re.sub(r"#+\s", "", topic)
            if not words:
                continue
            topic_total = sum(w_token)

            if topic_total <= budget:
                if current_tokens + topic_total <= budget:
                    if not current_parts: # this topic opens the group
                        current_head_topic = topic
                        current_head_words = words
                        current_head_wtok = w_token
                    current_parts.append(' '.join(words))
                    current_titles.append(topic)
                    current_tokens += topic_total
                else:
                    flush_merged()
                    current_parts = [' '.join(words)]
                    current_titles = [topic]
                    current_tokens = topic_total
                    current_head_topic, current_head_words, current_head_wtok = topic, words, w_token
            else:
                flush_merged()
                split_big_topic(words, w_token, topic)

        flush_merged()

        # Phase 2: append look-ahead overlap - each chunk's tail previews the
        # NEXT chunk's head. The next chunk is untouched.
        def head_overlap_text(head_topic, words, w_token, n_tokens):
            if not words or n_tokens <= 0:
                return ""
            total, idx = 0, 0
            while idx < len(words) and total < n_tokens:
                total += w_token[idx]
                idx += 1
            preview = " ".join(words[:idx])
            return preview
            # return " ".join(words[:idx])
        
        final_chunks = []
        for idx, r in enumerate(raw):
            body = r["body"]
            if overlap_tokens > 0 and idx + 1 < len(raw):
                nxt = raw[idx + 1]
                preview = head_overlap_text(nxt["head_topic"], nxt["head_words"], nxt["head_wtok"], overlap_tokens)
                if preview:
                    body = body + " " + preview
            final_chunks.append(Chunk(
                id=generate_sha256_hash(body), title=r["title"],
                source=body, cleaned_text=preprocess_text(body)
            ))

        return final_chunks

    def execute_llm_step(self, step_name, func, *args, retries=2, wait_time=10, **kwargs):
        """
        executes an LLM step with retry. If all retries fail, skips the step
        instead of breaking the program.
        Returns:
            result if successful
            None if skipped
        """
        for attempt in range(retries + 1):
            try:
                return func(*args, **kwargs)
            except RETRYABLE_EXCEPTIONS as e:
                if attempt < retries:
                    logger.info("Retrying in %d seconds (%d/%d)...", wait_time, attempt + 1, retries)
                    time.sleep(wait_time)
                else:
                    logger.error("%s failed after %d attempts", step_name, retries + 1)
                    logger.warning("Skipping %s", step_name)
                    return None
            except Exception:
                logger.exception("Unexpected error during %s", step_name)
                logger.warning("Skipping this step")
                return None
            
    def _update_chunk_entities(self, chunk: Chunk, previous_high_level = None, generate_high_level = False):
        def call_llm():
            if generate_high_level:
                formatted_context = ('\n'.join(f'- {fact}' for fact in previous_high_level) if previous_high_level else 'none (first chunk, or no prior asset context)')
                result = self._llm.with_structured_output(Output).invoke(
                    [
                        SystemMessage(
                            content=chunk_entities_extraction_prompt.format(
                                text_chunk=chunk.source,
                                previous_high_level=formatted_context,
                            )
                        )
                    ]
                )
                print(f'\nhigh_level_summary: {result.high_level_summary}\n')
                return result
            else:
                return self._llm2.with_structured_output(OutputBasic).invoke([
                    SystemMessage(content=chunk_entities_extraction_prompt_basic.format(
                        text_chunk=chunk.source,
                    ))
                ])

        response = self.execute_llm_step(f'entity extraction for chunk {chunk.id}', call_llm)
        if response is None:
            return None

        chunk.summary = response.summary
        chunk.questions = response.questions
        chunk.high_level_summary = getattr(response, 'high_level_summary', None)
        chunk.keywords = []
        return chunk
    
    def generate_chunks(self, document_path = None, *, document: Document = None, file_bytes = None, file_name = None):

        if document is None:
            with Timer('Extracting document content'):
                # document = self.extract_content(file_bytes=file_bytes, file_name=file_name)
                document = self.extract_content(filepath=document_path)
        with Timer("Chunking document"):
            text_chunks, text_tokens, text_topics = self._title_based_chunking(document.source)
            assert len(text_chunks) == len(text_tokens) == len(text_topics)        
            chunk_list = self._adjust_token_size(text_chunks, text_tokens, text_topics)
            document.chunks = chunk_list
        # with Timer("Generating chunk's entities"):
        #     document.chunks = list(map(self._update_chunk_entities, chunk_list))
        
        return document
