from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    """
    Load a single PDF file.

    Args:
        pdf_path (str): Path of the PDF file.

    Returns:
        List of Document objects
    """

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


# Test PDF loading
if __name__ == "__main__":

    pdf_path = r"C:\Users\ASUS\Downloads\Employee-Handbook.pdf"

    documents = load_pdf(pdf_path)

    print(f"Total Pages: {len(documents)}")

    print("\nFirst page content:")
    print(documents[0].page_content[:500])