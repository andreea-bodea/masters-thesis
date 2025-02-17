# BMZ publicbody = 95 https://fragdenstaat.de/api/v1/document/?publicbody=95
# status = successful https://fragdenstaat.de/api/v1/document/?publicbody=95&status=successful
# https://fragdenstaat.de/behoerde/95/bundesministerium-fur-wirtschaftliche-zusammenarbeit-und-entwicklung/


import asyncio
import random
from pathlib import Path

from requests import get
from tqdm.auto import tqdm

from cache import cache
from backend.pdf_to_text import convert_pdf_to_text

get = cache(get)

async def download_documents():
    # attachments = get("https://fragdenstaat.de/api/v1/attachment/").json()["objects"] # does not have publicbody field 
    documents = get("https://fragdenstaat.de/api/v1/document").json()["objects"]
    attachments = [
        get(
            f"https://fragdenstaat.de/api/v1/document/?limit=50&offset={10*i}"
            # f"https://fragdenstaat.de/api/v1/attachment/?limit=50&offset={10*i}" 
        ).json()["objects"]
        for i in range(10)
    ]
    attachments = [a for s in attachments for a in s]
    urls = [attachment["file_url"] for attachment in attachments]
    random.seed(0)
    urls = random.sample(urls, 30)
    files = [get(url).content for url in urls]
    documents = Path("documents")
    documents.mkdir(exist_ok=True)
    print(f"Files are being saved to: {documents.absolute()}")
    for i, file in enumerate(tqdm(files, desc="Downloading Files")):
        # save pdf
        (documents / f"{i}.pdf").open("wb").write(file)
        # convert pdf to text
        #text = await convert_pdf_to_text(file, i)
        # save text
        #(documents / f"{i}.txt").open("w").write(text)

if __name__ == "__main__":
    asyncio.run(download_documents())