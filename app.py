import os
import pandas as pd

from dotenv import load_dotenv
from pandasai import SmartDataframe
from sqlalchemy import create_engine, text
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def init_db(user, password, host, port, db_name):
    db_uri = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}'
    return SQLDatabase.from_uri(db_uri)


def get_sql_query_chain(db):
    prompt_template = """
        You are an expert in MySQL. You have been given a SQL database about social media data whose schema is 
        given below. Your task is to write a syntactically correct SQL query based on this schema of the database and 
        the question asked by the user. Properly learn all the columns in the database. Do not include any other 
        commands except the SELECT command. Return only the SQL query to the asked question without any other extra 
        text, also do not include the SQL query in ```.
        
        The schema of the database:
        <SCHEMA>{schema}</SCHEMA>
        
        Example 1 -
        Question: Who are the most followed users in the social media platform?
        SQL query: SELECT u.username, COUNT(*) AS followers_count
        FROM follows f
        INNER JOIN users u ON f.followee_id = u.user_id
        GROUP BY u.username
        ORDER BY followers_count DESC;

        
        Example 2 -
        Question: Find all posts containing the hashtag ' #followme'
        SQL query: SELECT p.caption, p.created_at
        FROM post p
        INNER JOIN post_tags pt ON p.post_id = pt.post_id
        INNER JOIN hashtags h ON pt.hashtag_id = h.hashtag_id
        WHERE h.hashtag_name = ' #followme';

        Example 3 -
        Question: Find users who never comment
        SQL query: SELECT user_id, username AS ‘User Never Comment’
        FROM users
        WHERE user_id NOT IN (SELECT user_id FROM comments);
        
        Similarly, complete this
        
        Question: {question}
        SQL query: 
    """

    prompt = PromptTemplate.from_template(prompt_template)

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.08)

    return (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info()) |
        prompt |
        llm |
        StrOutputParser()
    )


def get_results(question, db):
    prompt_template = """
        You are an expert in MySQL. You have been given a SQL database about social media data whose schema 
        is given below. Given the user's question, the SQL query and the SQL response, generate a natural language 
        response for the user.
        
        The schema of the database:
        <SCHEMA>{schema}</SCHEMA>
        
        User's question: {question}
        SQL query: {sql_query}
        SQL response: {sql_response}
        Natural language response:
    """

    prompt = PromptTemplate.from_template(prompt_template)

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.3)

    global sql_query
    global sql_response
    sql_query = get_sql_query_chain(db).invoke({'question': question})
    sql_response = db.run(sql_query)

    chain = (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info()) |
        prompt |
        llm |
        StrOutputParser()
    )

    return chain.invoke({'question': question, 'sql_query': sql_query, 'sql_response': sql_response})


# def get_dv_rec(db, query, response):
#     prompt_template = """
#         You have been given a MySQL database about social media whose schema is given below. You have also been a SQL
#         query and it SQL response. Based on the given data, recommend the most useful data visualisation for the
#         response generated. Recommend only that visualisation that can be made using the columns selected in the SQL
#         query, nothing else. Return only the one visualisation with the variables being used, nothing extra.
#
#         The schema of the database:
#         <SCHEMA>{schema}</SCHEMA>
#
#         SQL query: {sql_query}
#         SQL response: {sql_response}
#         Data visualisations:
#     """
#
#     prompt = PromptTemplate.from_template(prompt_template)
#
#     repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#     llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)
#
#     chain = (
#             RunnablePassthrough.assign(schema=lambda _: db.get_table_info()) |
#             prompt |
#             llm |
#             StrOutputParser()
#     )
#
#     return chain.invoke({'sql_query': query, 'sql_response': response})


# def sentiment_analysis(comments):
#     prompt_template = """
#         You are an expert linguist, who is good at classifying social media posts and comments. You have been given
#         some comments or post captions from a social media. Classify each of them into neutral, happy, sad, funny or
#         informative. Translate them into english if they are not already. Return only the text and the classification
#         associated with it in points. Do not return any code, please.
#
#         Example 1 -
#         Comment: aag lga di h bss fire extinguisher bulana pdega
#         Output: aag lga di h bss fire extinguisher bulana pdega: funny
#
#         Example 2 -
#         Comment: nice man !! loved it
#         Output: nice man !! loved it: happy
#
#         The captions/comments:
#         {comments}
#     """
#
#     prompt = PromptTemplate.from_template(prompt_template)
#
#     repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#     llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)
#
#     chain = (
#             RunnablePassthrough.assign(schema=lambda _: db.get_table_info()) |
#             prompt |
#             llm |
#             StrOutputParser()
#     )
#
#     return chain.invoke({'comments': comments})

def data_qna(df, question):
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)

    sdf = SmartDataframe(df, config={'llm': llm})

    print(sdf.chat(question))


load_dotenv()

user = os.environ.get('USER')
password = os.environ.get('PASSWORD')
host = os.environ.get('HOST')
port = os.environ.get('PORT')
db_name = os.environ.get('DB_NAME')

db = init_db(user=user, password=password, host=host, port=port, db_name=db_name)

sql_query = ''
sql_response = ''

results = get_results(question='get the top 10 users with posts having the most likes', db=db)

print(results)
print()
print(f'SQL query: {sql_query}')
print()

engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}').connect()
df = pd.read_sql(text(sql_query), engine)
print(df)

if df.empty is False:
    cont = int(input('Wish to continue? 1. Yes 2. No\n'))
    if cont == 1:
        while True:
            question = input('Ask questions to your data or generate graphs and charts\n')
            if question == '/quit':
                break
            data_qna(df, question)

# rec = get_dv_rec(db=db, query=sql_query, response=sql_response)
# print(rec)

# sentiments = sentiment_analysis(sql_response)
# print(sentiments)
