from PIL import Image
import os
import base64
import mimetypes

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI()


def encode_image(image_path):

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    return base64_image


def extract_image_content(image_path):

    base64_image = encode_image(image_path)

    mime_type, _ = mimetypes.guess_type(image_path)

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
If there is a table, extract headers, rows and values.
If there is a chart, describe clearly visible labels, trends and values.
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


def load_image(path):

    image = Image.open(path)

    width, height = image.size

    result = extract_image_content(path)

    document = {
        "content": result,
        "metadata": {
            "source": os.path.basename(path),
            "file_type": image.format.lower(),
            "content_type": "image",
            "width": width,
            "height": height,
            "image_path": path
        }
    }

    return [document]


def load_image_folder(folder_path):

    all_documents = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            file_path = os.path.join(folder_path, filename)

            documents = load_image(file_path)

            all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":
    image_folder = "data/img_data"

    documents = load_image_folder(image_folder)

    print("Number of image documents:", len(documents))
    print("Image documents:")

    for document in documents:
        print(document["metadata"])
        print(document["content"])
        print("----------------------")