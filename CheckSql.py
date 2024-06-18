import mysql.connector

#specify your own databases here
def search(str):
    db_config = {
        'host': '******',
        'port': '****',
        'user': '****',
        'password': '*****,
        'database': '*****'
    }

    conn = mysql.connector.connect(**db_config)

    cursor = conn.cursor()
    cursor.execute(str)
    fish = cursor.fetchone()
    return fish[0]


