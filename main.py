from loaders.pdf_loader_and_extractor import load_pdf_folder
from loaders.txt_loader import load_txt_folder
from loaders.docx_loader_and_extractor import load_docx_folder
from loaders.xlsx_loader_and_extractor import load_xlsx_folder
from loaders.img_loader_and_extractor import load_image_folder

from chunker import chunk_documents

from collections import Counter


pdf_documents = load_pdf_folder("data/pdf_data")
txt_documents = load_txt_folder("data/txt_data")
docx_documents = load_docx_folder("data/docx_data")
xlsx_documents = load_xlsx_folder("data/xlsx_data")
image_documents = load_image_folder("data/img_data")


all_documents = (pdf_documents+ txt_documents+ docx_documents+ xlsx_documents+ image_documents)


pdf_types = Counter(
    document["metadata"]["content_type"]
    for document in pdf_documents
)

print("PDF:", len(pdf_documents))
print("PDF breakdown:", pdf_types)
print("TXT:", len(txt_documents))
print("DOCX:", len(docx_documents))
print("XLSX:", len(xlsx_documents))
print("Images:", len(image_documents))
print("TOTAL:", len(all_documents))


# CHUNKING
chunks = chunk_documents(all_documents)

print("Documents:", len(all_documents))
print("Chunks:", len(chunks))


chunk_types = Counter(
    chunk["metadata"]["content_type"]
    for chunk in chunks
)

print("Chunk breakdown:", chunk_types)

### EMBEDDINGS
from embeddings import create_embeddings

chunks = create_embeddings(chunks)

##  VECTOR DB
from vector_db import create_vector_db

collection = create_vector_db(chunks)