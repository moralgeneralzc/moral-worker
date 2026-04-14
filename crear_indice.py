import psycopg2

conn = psycopg2.connect(
    "postgresql://postgres:Sucre1552..@db.xhnurftxuinlycmbsgij.supabase.co:5432/postgres"
)
conn.autocommit = True
cur = conn.cursor()

print("Creando indice HNSW, espera unos minutos...")
cur.execute("SET statement_timeout = '600000'")
cur.execute("""
    CREATE INDEX IF NOT EXISTS fragmentos_embedding_hnsw_idx
    ON fragmentos
    USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64)
""")
print("Indice creado exitosamente!")
cur.close()
conn.close()