import os
import openai
#specify your API KEY here
os.environ["OPENAI_API_KEY"] = "****"
openai.api_key = os.environ["OPENAI_API_KEY"]
from IPython.display import Markdown, display
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)

#specify the link to databse 
url_str = "**************"

engine = create_engine(
    url_str,
    echo = True
)
#echo=True: Enables the logging of all SQL statements.
#This is useful for debugging to see exactly what SQL is being executed and can help in identifying issues with queries
#or understanding the flow of database interactions.


metadata_obj = MetaData()   

# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)

metadata_obj.create_all(engine)

from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
sql_database = SQLDatabase(engine, include_tables=["city_stats"])
sql_database = SQLDatabase(engine, include_tables=["city_stats"])

from sqlalchemy import insert

rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]

for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)


# view current table
stmt = select(
    city_stats_table.c.city_name,
    city_stats_table.c.population,
    city_stats_table.c.country,
).select_from(city_stats_table)

with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()
    print(results)

from sqlalchemy import text

with engine.connect() as con:
    rows = con.execute(text("SELECT city_name from city_stats"))
    for row in rows:
        print(row)

from llama_index.core.query_engine import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database, tables=["city_stats"], llm=llm
)

query_str = "Which city has the highest population?"

response = query_engine.query(query_str)

display(Markdown(f"{response}"))


from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import VectorStoreIndex

# set Logging to DEBUG for more detailed outputs
table_node_mapping = SQLTableNodeMapping(sql_database)

table_schema_objs = [
    (SQLTableSchema(table_name="city_stats"))
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)

query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=1)
)

response = query_engine.query("Which city has the highest population?")
display(Markdown(f"{response}"))


response.metadata["result"]

# manually set context text
city_stats_text = (
    "This table gives information regarding the population and country of a"
    " given city.\nThe user will query with codewords, where 'foo' corresponds"
    " to population and 'bar'corresponds to city."
)

table_node_mapping = SQLTableNodeMapping(sql_database)

table_schema_objs = [
    (SQLTableSchema(table_name="city_stats", context_str=city_stats_text))
]


from llama_index.core.retrievers import NLSQLRetriever

# default retrieval (return_raw=True)
nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=["city_stats"], return_raw=True
)

results = nl_sql_retriever.retrieve(
    "Return the top 5 cities (along with their populations) with the highest population."
)

from llama_index.core.response.notebook_utils import display_source_node

for n in results:
    display_source_node(n)

# default retrieval (return_raw=False)
nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=["city_stats"], return_raw=False
)

results = nl_sql_retriever.retrieve(
    "Return the top 5 cities (along with their populations) with the highest population."
)

# NOTE: all the content is in the metadata
for n in results:
    display_source_node(n, show_source_metadata=True)

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(nl_sql_retriever)

response = query_engine.query(
    "Return the top 5 cities (along with their populations) with the highest population."
)

print(str(response))
