FROM rbergm/postbound-sigmod25:latest

WORKDIR /sigmod25
RUN git clone https://github.com/db-tu-dresden/SIGMOD25-OptimizerTutorial.git /sigmod25 && \
    cp /sigmod25/.psycopg_connection_stats.sample /sigmod25/.psycopg_connection && \
    source /postbound/pb-venv/bin/activate && \
    pip install sentence-transformers scikit-learn
