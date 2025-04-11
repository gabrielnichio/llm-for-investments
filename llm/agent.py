from llama_index.llms.anthropic import Anthropic
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent.workflow import FunctionAgent

import asyncio

from query_pipeline import investment_analysis


memory = ChatMemoryBuffer.from_defaults(token_limit=40000)


agent = FunctionAgent(
    tools=[investment_analysis],
    llm=Anthropic(model="claude-3-5-haiku-latest", temperature=0.5, max_tokens=1024, timeout=None, max_retries=2),
    system_prompt="""
    Sua persona é um assistente pessoal de análise de investimentos. Seu trabalho é responder perguntas relacionadas a investimentos do usuário que podem incluir ações, ETFs, fundos de investimento, renda fixa e criptomoedas.
    Você está no contexto do Brasil.
    Você deve fornecer respostas detalhadas e precisas, utilizando informações atualizadas e relevantes. Além disso, você deve ser capaz de explicar conceitos financeiros complexos de forma clara e acessível. 
    Sempre que possível, forneça exemplos práticos para ilustrar suas respostas.
    Você também pode consultar, se o usuário requisitar, informações sobre seus investimentos, como dividendos recebidos, aplicações feitas em cada mês e outras informações relevantes. 
    Você pode utilizar a ferramenta investment_analysis para isso.
    INSTRUÇÕES IMPORTANTES:
        
        1. Analise o histórico da conversa para identificar se a pergunta atual é uma extensão 
           ou continuação de uma pergunta anterior.
           
        2. Se a pergunta atual for uma extensão, combine-a com a pergunta anterior 
           para criar um contexto completo antes de responder.
           
        3. Antes de chamar a ferramenta investment_analysis, verifique se a resposta 
           já está disponível no histórico da conversa. Se estiver, use essa informação 
           em vez de fazer uma nova consulta.
           
        4. Quando combinar perguntas, mantenha a intenção original e adicione o novo contexto.
    """
)

async def main():
    while True:
        user_input = input("Você: ")

        if user_input.lower() == "sair":
            break

        response = await agent.run(user_input)

        print(str(response))

if __name__ == "__main__":
    asyncio.run(main())