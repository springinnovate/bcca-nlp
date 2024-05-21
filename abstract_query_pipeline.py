from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import logging
import numpy
import os
import pickle
import re
import sys
import textwrap

from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import faiss
import fitz
import spacy
import torch
import tiktoken

GPT_MODEL = 'gpt-4o'
#GPT_MODEL = 'gpt-3.5-turbo'
ENCODING = tiktoken.encoding_for_model(GPT_MODEL)

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.WARN)
logging.getLogger('PyPDF2._reader').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)

GPT_MODEL = 'gpt-4o'
#GPT_MODEL = 'gpt-3.5-turbo'
TOP_K = 100

# how big of a window to build around sentences
SLIDING_WINDOW_SENTENCE_SIZE = 10
SENTENCE_WINDOW_OVERLAP = 2
SLIDING_WINDOW_CHARACTER_SIZE = 500
BODY_TAG = 'body'
CITATION_TAG = 'citation'

DATA_STORE = [
    "data/*.pdf"
]

CACHE_DIR = 'llm_cache'
for dirpath in [CACHE_DIR]:
    os.makedirs(dirpath, exist_ok=True)


def trim_conversation(context, max_tokens=4096):
    # Tokenizer should be used to count tokens accurately; this is a simplified example
    total_tokens = len(ENCODING.encode(context))
    while total_tokens > max_tokens:
        overshoot = max_tokens - total_tokens
        context = ' '.join(context.split()[:-overshoot//2])
        print(len(context))
        total_tokens = len(ENCODING.encode(context))
    return context

# Example function to concatenate document contents and generate a hash
def generate_hash(documents):
    concatenated = ''.join(documents)
    hash_object = hashlib.sha256(concatenated.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    return hash_hex


spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(pdf_path):
    sentence_windows_per_page = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            print(f'processing page {page_num} {os.path.basename(pdf_path)}')
            raw_text = page.get_text()
            clean_text = raw_text.replace(' \n', ' ').replace('\n', ' ')
            sentence_windows = split_into_sentence_windows(page_num, clean_text)
            sentence_windows_per_page.append((page_num, sentence_windows))
    return sentence_windows_per_page


def split_into_sentence_windows(page_num, text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    current_sentence_offset = 0
    sentence_windows = []
    while current_sentence_offset < len(sentences):
        working_window = ' '.join(sentences[current_sentence_offset:current_sentence_offset + SLIDING_WINDOW_SENTENCE_SIZE])
        current_sentence_offset += SLIDING_WINDOW_SENTENCE_SIZE
        while len(working_window) < SLIDING_WINDOW_CHARACTER_SIZE and current_sentence_offset < len(sentences):
            working_window += ' ' + sentences[current_sentence_offset]
            current_sentence_offset += 1
        sentence_windows.append(working_window)
        if current_sentence_offset != len(sentences):
            current_sentence_offset -= SENTENCE_WINDOW_OVERLAP
    return sentence_windows


def parse_file(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        raise NotImplementedError(f'No parser for {file_path}')


def save_embeddings(documents, model, filename):
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    document_embeddings_np = document_embeddings.cpu().numpy()
    numpy.save(filename, document_embeddings_np)


def load_embeddings(filename):
    document_embeddings_np = numpy.load(filename)
    document_embeddings = torch.tensor(document_embeddings_np)
    return document_embeddings


def main():
    file_paths = [
        file_path
        for file_pattern in DATA_STORE
        for file_path in glob.glob(file_pattern)]

    file_hash = generate_hash(file_paths)
    expanded_sentence_window_path = os.path.join(CACHE_DIR, f'{file_hash}.pkl')
    fiass_path = os.path.join(CACHE_DIR, f'{file_hash}.faiss')

    from sentence_transformers import SentenceTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2').to(device)
    qa_model_list = []
    for model_id in [
            #'FacebookAI/xlm-roberta-large',
            #'timpal0l/mdeberta-v3-base-squad2',
            #'deepset/bert-large-uncased-whole-word-masking-squad2',
            #'distilbert/distilbert-base-cased-distilled-squad',
            #'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad',
            #'allenai/scibert_scivocab_uncased',
            'deepset/roberta-base-squad2',
            ]:
        qa_model_list.append((
            model_id,
            None))

    if os.path.exists(expanded_sentence_window_path):
        with open(expanded_sentence_window_path, 'rb') as file:
            expanded_sentence_windows = pickle.load(file)
        file_path_list, page_num_list, sentence_windows = zip(
            *expanded_sentence_windows)
        document_distance_index = faiss.read_index(fiass_path)
    else:
        article_list = []
        for file_path in file_paths:
            article_list.append((
                os.path.basename(file_path),
                parse_file(file_path)))

        expanded_sentence_windows = [
            (file_path, page_num, sentence_window)
            for file_path, sentence_windows_per_page in article_list
            for page_num, sentence_windows in sentence_windows_per_page
            for sentence_window in sentence_windows]
        with open(expanded_sentence_window_path, 'wb') as file:
            pickle.dump(expanded_sentence_windows, file)
        file_path_list, page_num_list, sentence_windows = zip(
            *expanded_sentence_windows)

        LOGGER.debug('embedding')
        document_embeddings = embedding_model.encode(
            sentence_windows, convert_to_tensor=True)
        LOGGER.debug('indexing')
        document_distance_index = faiss.IndexFlatL2(document_embeddings.shape[1])
        document_distance_index.add(document_embeddings.cpu().numpy())
        faiss.write_index(document_distance_index, fiass_path)

    def query_index(question):
        question_embedding = embedding_model.encode(
            question, convert_to_tensor=True).cpu().numpy()

        # Ensure the question_embedding is 2D
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.reshape(1, -1)

        # Retrieve the most similar documents
        distances, indices = document_distance_index.search(
            question_embedding, TOP_K)
        print(f'****************** {distances}, {indices}')
        retrieved_windows = [sentence_windows[idx] for idx in indices[0]]
        relevant_page_numbers = [page_num_list[idx] for idx in indices[0]]
        relevant_files = [file_path_list[idx] for idx in indices[0]]

        answers = []

        for qa_model_id, qa_model in qa_model_list:
            print(qa_model_id)
            qa_model = pipeline('question-answering', model=model_id, device=device)
            for context, page_number, filename in zip(
                    retrieved_windows, relevant_page_numbers, relevant_files):
                answer = qa_model(question=question, context=context)
                answers.append((answer['answer'], answer['score'], qa_model_id, context, page_number, filename))
            del qa_model

        response = question
        for answer, score, qa_model_id, context, page_number, filename in sorted(
                answers, key=lambda x: x[1], reverse=False):
            response += (
                f'\n\n  * "{question}"?\n\tAnswer: "{answer}" ({score:.3f} (out of 1.0) - {qa_model_id})'
                f'\n\t  -- context from page {page_number} of "{filename}":'
                f'\n\t\t{context}')
        return response

    def answer_question_with_gpt(question):
        # Encode the question
        from openai import OpenAI
        client = OpenAI()

        question_embedding = embedding_model.encode(
            question, convert_to_tensor=True).cpu().numpy()

        # Ensure the question_embedding is 2D
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.reshape(1, -1)

        distances, indices = document_distance_index.search(
            question_embedding, TOP_K)

        retrieved_windows = [sentence_windows[idx] for idx in indices[0]]
        relevant_page_numbers = [page_num_list[idx] for idx in indices[0]]
        relevant_files = [file_path_list[idx] for idx in indices[0]]

        # Concatenate the retrieved documents to form the context
        context = " ".join([
            f'reference index: {index}", context: {context}\n\n'
            for index, (filename, page_number, context) in enumerate(zip(
                relevant_files, relevant_page_numbers, retrieved_windows))])

        context = trim_conversation(context, max_tokens=20000)
        context_counts = context.count("context: ")
        stream = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are given a set of filename/page number/text snippets. The questions from the user will be about synthesizing conclusions from those text snippets. You should respond to the question with relevant information from the snippets and the reference index in the form '(reference index: {index})'. If you do not have enough information to answer, say so. Do not make up any information."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\nAnswer:"}
            ],
            stream=True,
            max_tokens=4000  # Adjust the number of tokens to get a longer answer
        )
        response = f'From {context_counts} references to the texts -- '
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        index_matches = re.findall(r'\(reference index: \{?(\d+(?:, \d+)*?)\}?\)', response)
        parsed_indexes = set([
            int(num) for match in index_matches for num in match.split(', ')])
        wrapper = textwrap.TextWrapper(
            width=80,
            initial_indent='    ',  # Initial indent for the first line
            subsequent_indent='    '  # Indent for all subsequent lines
        )
        wrapped_response = wrapper.fill(response)

        wrapper = textwrap.TextWrapper(
            width=70,
            initial_indent='       ',  # Initial indent for the first line
            subsequent_indent='       '  # Indent for all subsequent lines
        )

        processed_indexes = set()
        for index in sorted(parsed_indexes):
            index = int(index)
            if index in processed_indexes:
                continue
            processed_indexes.add(index)
            wrapped_response += '\n\n' + wrapper.fill(
                f'reference index {index}. "{relevant_files[index]}" pg{relevant_page_numbers[index]}: '
                f'{retrieved_windows[index]}')
        return wrapped_response

    while True:
        question = input("\nEnter your question (or type 'exit' to exit): ")
        if question.lower() == 'exit':
            break
        if question.strip() == '':
            continue
        #response = query_index(question)
        response = answer_question_with_gpt(question)
        print('Answer:\n' + response)


if __name__ == '__main__':
    load_dotenv()
    main()
