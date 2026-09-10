from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)

    chunks = []

    for document in documents:

        content = document["content"]
        metadata = document["metadata"]

        content_type = metadata.get("content_type")

        if content_type == "text":

            text_chunks = splitter.split_text(content)

            for chunk_index, chunk_text in enumerate(text_chunks, start=1):

                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index

                chunk = {
                    "content": chunk_text,
                    "metadata": chunk_metadata
                }

                chunks.append(chunk)

        else:

            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = 1

            chunk = {
                "content": content,
                "metadata": chunk_metadata
            }

            chunks.append(chunk)

    return chunks