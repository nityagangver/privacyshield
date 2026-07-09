FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m -u 1000 user && chown -R user:user /app
USER 1000
EXPOSE 7860
ENV PYTHONUNBUFFERED=1
RUN mkdir -p /app/graphs /app/models
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
