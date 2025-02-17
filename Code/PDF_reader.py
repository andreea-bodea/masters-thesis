import base64
import io
import os
from functools import lru_cache

import pymupdf
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from tqdm.asyncio import tqdm

load_dotenv()

openai_model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=1.0,
)

model = openai_model

async def pdf_page_to_base64(buffer: bytes, page_num: int) -> str:
    document = pymupdf.open("pdf", buffer)
    page = document.load_page(page_num)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


@lru_cache(maxsize=128)
async def pdf_page_to_text(buffer: bytes, page_num: int) -> str:
    base64_image = await pdf_page_to_base64(buffer, page_num)
    response = model.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Return the content from the PDF page in Markdown format. Do not use a ``` code block but just return the Markdown text. For redacted text, use █ characters, if applicable.",
                    },
                    # see https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            )
        ]
    )
    return response.content


@lru_cache(maxsize=128)
async def convert_pdf_to_text(buffer: bytes, i: int | None = None) -> str:
    num_pages = pymupdf.open("pdf", buffer).page_count
    tasks = [pdf_page_to_text(buffer, page_num) for page_num in range(num_pages)]
    desc = "Processing PDF Pages"
    if i is not None:
        desc += f" for document {i}"
    results = await tqdm.gather(*tasks, total=num_pages, desc=desc)
    text = "\n\n\n".join(results)
    return text
