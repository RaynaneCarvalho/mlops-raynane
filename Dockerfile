FROM python:3.11-slim

WORKDIR /app

# dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# código
COPY app ./app
COPY models ./models/

# opcional: configurações visuais do Streamlit (tema)
COPY .streamlit ./.streamlit

# porta do Streamlit
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
