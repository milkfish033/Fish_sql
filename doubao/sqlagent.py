from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)

from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.settings import Settings
from custom_tools.DoubaoLLM import DoubaoLLM

from LocalEmbedding import LocalGTEBaseEmbedding

url_str = "*********"
#specify the link to databse here 

engine = create_engine(
        url_str,
        echo=True
    )
#echo=True: Enables the logging of all SQL statements.
#This is useful for debugging to see exactly what SQL is being executed and can help in identifying issues with queries
#or understanding the flow of database interactions.


llm = DoubaoLLM()
Settings.embed_model = LocalGTEBaseEmbedding()

sql_db = SQLDatabase(engine=engine, include_tables=['sys_account', 'sys_account_approve', 'user_info', 'organ_village_info', 'task_info'])
query_engine = NLSQLTableQueryEngine(sql_database=sql_db, llm=llm)

response = query_engine.query("统计枫桥镇有多少人？")
print(response)
