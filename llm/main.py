from assistant import Assistant
from tools.document_display import display_similar_documents, display_source_documents




if __name__ == "__main__":
    assistant = Assistant()
    llm = assistant.llm()
    
    while True:
        user_input = input("\nInsira sua pergunta ou digite 'sair' para encerrar: ")
        
        if user_input.lower() == "sair":
            print("Encerrando o sistema...")
            break
            
        # Recuperar embedding da consulta
        query_embeddings = assistant.get_embedding().get_text_embedding(user_input)
        chroma_collection = assistant.get_chroma_collection()
        
        display_similar_documents(chroma_collection, query_embeddings)
        
        print("\nGerando resposta...")
        response = llm.stream_chat(user_input)
        print(f"\nResposta: {response.print_response_stream()}")
        
        # MODIFICAÇÃO: Exibir os documentos utilizados na resposta
        display_source_documents(response)