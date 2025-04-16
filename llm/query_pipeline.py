import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from llama_index.core.query_pipeline import ( QueryPipeline as QP, Link, InputComponent )
from llama_index.experimental.query_engine.pandas import ( PandasInstructionParser )

from llama_index.core import PromptTemplate
from llama_index.llms.anthropic import Anthropic

df = pd.read_csv("../data/applications.csv")

def get_df_info(df: pd.DataFrame) -> str:
    """
    Get the information of the dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe to get the information from.
        
    Returns:
        str: The information of the dataframe.
    """
    infos = ""

    for column in df.columns:
        infos += f"{column}: {df[column].dtype}\n"

    return infos

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python related to stock investments.\n"
    "The dataframe contains the following columns:\n"
    "{df_info}\n"
    "'Cota' is the name of the stock, 'Valor' is the ammount of money that was applied in the stock, 'Data' is the date of the investment, 'Tipo' is the type of the record, 'Mes' is the month of the application and 'Ano' is the year of the application.\n"
    "'Valor' is not the quantity of stocks, but the amount of money that was applied in the stock.\n"
    "Each row represents a record of an investment made by the user. There can be more than one investment in a month.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)
response_synthesis_prompt_str = (
    "Your persona is a personal investment analysis assistant. Your job is to answer questions related to the user's investments."
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5), df_info=get_df_info(df)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = Anthropic(model="claude-3-5-haiku-latest", temperature=0, max_tokens=1024, timeout=None, max_retries=2)

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True
)

qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link(
            "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
        ),
        Link(
            "pandas_output_parser",
            "response_synthesis_prompt",
            dest_key="pandas_output",
        ),
    ]
)

qp.add_link("response_synthesis_prompt", "llm2")

def investment_analysis(question: str) -> str:
    """
    Get the user question about their own investments, turn into a Pandas query and return the result.
    
    Args:
        question (str): The user's question regarding their investments.
        
    Returns:
        str: The analysis result.
    """

    response = qp.run(
        query_str = question,
    
    )
    return response.message.content




