import psycopg2

def get_batch_id():
    conn = psycopg2.connect(
        host="postgres",       
        port=5432,
        dbname="airflow_db",      
        user="airflow",        
        password="airflow"     
    )
    cur = conn.cursor()
    cur.execute("SELECT val FROM variable WHERE key = 'batch_id'")
    val = cur.fetchone()[0]
    conn.close()
    return int(val)

if __name__ == "__main__":
    batch_id = get_batch_id()
    if batch_id == 1:
        print("batch_id == 1 → On exécute les scripts.")
        exit(0)
    else:
        print(f"batch_id = {batch_id} ≠ 1 → On skip.")
        exit(1)
