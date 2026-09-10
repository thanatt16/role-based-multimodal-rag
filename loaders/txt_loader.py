import os


def load_txt(path):

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    document = {
        "content": text,
        "metadata": {
            "source": os.path.basename(path),
            "file_type": "txt",
            "content_type": "text"
        }
    }

    return [document]


def load_txt_folder(folder_path):

    all_documents = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".txt"):

            file_path = os.path.join(folder_path, filename)

            documents = load_txt(file_path)

            all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":

    txt_folder = "data/txt_data"

    documents = load_txt_folder(txt_folder)

    print("Number of TXT documents:", len(documents))

    for document in documents:
        print(document["metadata"])
        print(document["content"][:200])
        print("--------------------")