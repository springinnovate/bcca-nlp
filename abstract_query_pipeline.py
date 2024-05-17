from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import logging
import numpy
import os
import pickle
import re
import sys

from nltk.tokenize import sent_tokenize
import faiss
import fitz
import torch
import spacy

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.WARN)
logging.getLogger('PyPDF2._reader').setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)

#GPT_MODEL = 'gpt-4o' # 'gpt-3.5-turbo'
#ENCODING = tiktoken.encoding_for_model(GPT_MODEL)
TOP_K = 10

# how big of a window to build around sentences
SLIDING_WINDOW_SIZE = 10
BODY_TAG = 'body'
CITATION_TAG = 'citation'

DATA_STORE = [
    "data/*.pdf"
]

CACHE_DIR = 'llm_cache'
for dirpath in [CACHE_DIR]:
    os.makedirs(dirpath, exist_ok=True)



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
            sentence_windows = split_into_sentence_windows(clean_text)
            if page_num > 5:
                break
            sentence_windows_per_page.append((page_num, sentence_windows))
    return sentence_windows_per_page


def split_into_sentence_windows(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    current_offset = 0
    sentence_window = []
    while current_offset < len(sentences):
        sentence_window.append(' '.join(sentences[
            current_offset:current_offset + SLIDING_WINDOW_SIZE]))

        current_offset += SLIDING_WINDOW_SIZE - 1
    return sentence_window


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

    if False and os.path.exists(expanded_sentence_window_path):
        with open(expanded_sentence_window_path, 'rb') as file:
            expanded_sentence_windows = pickle.load(file)
        file_path_list, page_num_list, sentence_windows = zip(
            *expanded_sentence_windows)
    else:
        article_list = []
        for file_path in file_paths:
            article_list.append((
                os.path.basename(file_path),
                parse_file(file_path)))
            break

        expanded_sentence_windows = [
            (file_path, page_num, sentence_window)
            for file_path, sentence_windows_per_page in article_list
            for page_num, sentence_windows in sentence_windows_per_page
            for sentence_window in sentence_windows]
        with open(expanded_sentence_window_path, 'wb') as file:
            pickle.dump(article_list, file)
        file_path_list, page_num_list, sentence_windows = zip(
            *expanded_sentence_windows)

        LOGGER.debug('embedding')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2').to(device)
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
        relevent_text = [sentence_windows[idx] for idx in indices[0]]
        relevent_page_numbers = [page_num_list[idx] for idx in indices[0]]
        relevant_files = [file_path_list[idx] for idx in indices[0]]

        return '\n   * '.join([
            f'{file_path}: {page_num} - \n\t{" ".join(sentence_window)}'
            for file_path, page_num, sentence_window in
            zip(relevent_text,
                relevent_page_numbers,
                relevant_files)])

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        if question.strip() == '':
            continue
        response = query_index(question)
        print(f"\n\n{response}")


if __name__ == '__main__':
    main()
