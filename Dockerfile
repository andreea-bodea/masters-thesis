FROM --platform=linux/amd64 python:3.10-slim
WORKDIR /app
RUN pip install spacy
RUN python -m spacy download en_core_web_md
COPY . .
RUN pip install --no-cache-dir -e .
EXPOSE 8501
ENV PYTHONUNBUFFERED=1
CMD ["streamlit", "run", "src/Demo/Streamlit_Enron_BBC.py", "--server.address=0.0.0.0", "--server.port=8501"]
