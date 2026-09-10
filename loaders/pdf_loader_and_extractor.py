import pymupdf
import os
import base64

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI()


def extract_image_content(image_bytes, image_extension):

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    if image_extension.lower() in ["jpg", "jpeg"]:
        mime_type = "image/jpeg"
    else:
        mime_type = f"image/{image_extension.lower()}"

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": """
Extract all useful information from this image for a RAG knowledge base.

Read all visible text accurately.
Preserve important numbers, names, labels and values exactly as shown.

If there is a table:
- Extract its headers, rows and values.

If there is a chart:
- Extract visible labels, axes and values.
- Describe clearly visible trends and relationships.
- Do not invent precise values that are not explicitly visible.

If there is a diagram:
- Describe its components and relationships.

Do not invent information that is not visible.
"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high"
                    }
                ]
            }
        ]
    )

    return response.output_text


def table_to_text(table_data):

    rows = []

    for row in table_data:

        values = []

        for value in row:

            if value is None:
                values.append("")
            else:
                values.append(str(value).strip())

        row_text = " | ".join(values)

        rows.append(row_text)

    return "\n".join(rows)


def load_pdf(path):

    doc = pymupdf.open(path)

    documents = []

    for page_number, page in enumerate(doc):

        # ==============================
        # TEXT
        # ==============================

        text = page.get_text()

        text_document = {
            "content": text,
            "metadata": {
                "source": os.path.basename(path),
                "file_type": "pdf",
                "content_type": "text",
                "page": page_number + 1
            }
        }

        documents.append(text_document)

        # ==============================
        # TABLES
        # ==============================

        tables = page.find_tables()

        print(
            f"{os.path.basename(path)} - "
            f"Page {page_number + 1}: "
            f"{len(tables.tables)} tables"
        )

        for table_index, table in enumerate(
            tables.tables,
            start=1
        ):

            table_data = table.extract()

            table_text = table_to_text(table_data)

            table_document = {
                "content": table_text,
                "metadata": {
                    "source": os.path.basename(path),
                    "file_type": "pdf",
                    "content_type": "table",
                    "page": page_number + 1,
                    "table_index": table_index
                }
            }

            documents.append(table_document)

        # ==============================
        # IMAGES
        # ==============================

        images = page.get_images(full=True)

        print(
            f"{os.path.basename(path)} - "
            f"Page {page_number + 1}: "
            f"{len(images)} images"
        )

        for image_index, image in enumerate(
            images,
            start=1
        ):

            xref = image[0]

            image_data = doc.extract_image(xref)

            image_bytes = image_data["image"]
            image_extension = image_data["ext"]

            vlm_result = extract_image_content(
                image_bytes,
                image_extension
            )

            image_document = {
                "content": vlm_result,
                "metadata": {
                    "source": os.path.basename(path),
                    "file_type": "pdf",
                    "content_type": "image",
                    "page": page_number + 1,
                    "image_index": image_index,
                    "xref": xref
                }
            }

            documents.append(image_document)

    return documents


def load_pdf_folder(folder_path):

    all_documents = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".pdf"):

            file_path = os.path.join(
                folder_path,
                filename
            )

            documents = load_pdf(file_path)

            all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":

    pdf_folder = r"E:\Machine_learning\AI\RAG\PROJECT\data\pdf_data"

    documents = load_pdf_folder(pdf_folder)

    print("\nNumber of documents:", len(documents))

    print("\nDocuments:")

    for document in documents:

        print(document["metadata"])

        print(document["content"][:300])

        print("----------------------")