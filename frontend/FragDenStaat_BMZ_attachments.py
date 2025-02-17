import requests
import json
import os

url = "https://fragdenstaat.de/api/v1/document/"
params = {
    "publicbody": 95,  # BMZ Public Body ID
    "status": "successful",  # Only successful requests
}
response = requests.get(url, params=params)

documents = response.json()
document_objects = documents.get("objects", [])
for doc_object in document_objects[:2]:
    print(json.dumps(doc_object, indent=4))

"""
# Extract the URLs
document_urls = [doc["file_url"] for doc in document_objects if "file_url" in doc]
print(document_urls)

# Use a directory within your home directory
save_dir = os.path.expanduser("~/FragDenStaat")
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Iterate over document_objects, not documents
for doc in document_objects:
    doc_url = doc["file_url"]
    doc_title = doc["title"].replace("/", "_")  # Ensure valid filename
    
    file_path = os.path.join(save_dir, f"{doc_title}.pdf")
    
    # Download the file
    doc_response = requests.get(doc_url)
    with open(file_path, "wb") as file:
        file.write(doc_response.content)

    print(f"Downloaded: {file_path}")"""