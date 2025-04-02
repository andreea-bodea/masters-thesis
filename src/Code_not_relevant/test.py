# Import the functions from Pinecone_LlamaIndex
from RAG.Pinecone_LlamaIndex import loadDataPinecone, getResponse

# Test loadDataPinecone function
def test_loadDataPinecone():
    try:
        loadDataPinecone(
            index_name="test",
            text="test document 4",
            file_name="test file for Andreea",
            file_hash="HASH HASH HASH ",
            text_type="text_with_pii TEST"
        )
        print("loadDataPinecone: Success")
    except Exception as e:
        print(f"loadDataPinecone: Failed with error {e}")

# Test getResponse function
def test_getResponse():
    try:
        response = getResponse(
            index_name="test", 
            question="What are the types of RAG systems?"
        )
        print("getResponse: Success")
        print("Response:", response)
    except Exception as e:
        print(f"getResponse: Failed with error {e}")

# Run tests
if __name__ == "__main__":
    test_loadDataPinecone()
    test_getResponse()