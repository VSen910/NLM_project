import os

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def init_db(user, password, host, port, db_name):
    db_uri = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}'
    return SQLDatabase.from_uri(db_uri)


def get_sql_query_chain(db):
    prompt_template = """
        You are an expert in MySQL. You have been given a SQL database about a bakery whose schema is given below.
        Your task is to write a SQL query based on this schema of the database and the question asked by the user.
        Do not include any other commands except the SELECT command. Return only the SQL query to the asked question
        without any other extra text, also do not include the SQL query in ```.
        
        The schema of the database:
        {schema}
        
        Example 1 -
        Question: Who is the customer whose customer id is 11
        SQL query: SELECT * FROM customers WHERE cid = 11;
        
        Example 2 -
        Question: What are the names of the customers who have more than 10 receipts?
        SQL query: SELECT * FROM customers WHERE cid IN (SELECT CustomerId FROM receipts GROUP BY CustomerId 
        HAVING COUNT(ReceiptNumber) > 10);
        
        Similarly, complete this
        
        Question: {question}
        SQL query: 
    """

    prompt = PromptTemplate.from_template(prompt_template)

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)

    return (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info()) |
        prompt |
        llm |
        StrOutputParser()
    )

def get_results(question, db):
    prompt_template = """
        You are an expert in MySQL. You have been given a SQL database about a bakery whose schema is given below.
        Given the user's question, the SQL query and the SQL response, generate a natural language response for the user.
        
        The schema of the database:
        {schema}
        
        User's question: {question}
        SQL query: {sql_query}
        SQL response: {sql_response}
        Natural language response:
    """

    prompt = PromptTemplate.from_template(prompt_template)

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.3)

    sql_query = get_sql_query_chain(db)

    chain = (
        RunnablePassthrough.assign(sql_query=sql_query).assign(
            schema=lambda _: db.get_table_info(),
            sql_response=lambda vars: db.run(vars['sql_query'])
        ) |
        prompt |
        llm |
        StrOutputParser()
    )

    return chain.invoke({'question': question})


load_dotenv()

db = init_db(user=os.environ.get('USER'), password=os.environ.get('PASSWORD'), host=os.environ.get('HOST'),
        port=os.environ.get('PORT'), db_name=os.environ.get('DB_NAME'))

# response = get_sql_query_chain(db).invoke({
#     'question': 'who are the customers who bought something on the 9 oct 2007'
# })
# print(response)

# print(get_sql_query_chain(db).invoke({
#     'question': 'who are the unique customers who bought something on the 9-Oct-2007'
#     })
# )

response = get_results(question='who are the unique customers who bought something on the 9-Oct-2007', db=db)
print(response)
