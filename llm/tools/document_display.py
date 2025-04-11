def display_source_documents(response):
    """Exibe os documentos fonte utilizados para gerar a resposta."""
    print("\n" + "=" * 30)
    print("DOCUMENTOS UTILIZADOS PARA GERAR A RESPOSTA:")
    print("=" * 30)
    
    if hasattr(response, 'source_nodes') and response.source_nodes:
        for i, source_node in enumerate(response.source_nodes):
            print(f"\nDocumento fonte #{i+1}:")
            print(f"ID: {source_node.node.node_id}")
            print(f"Score: {source_node.score:.4f}")
            
            # Exibir o texto do documento (limitado para não sobrecarregar o terminal)
            text = source_node.node.text
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"Conteúdo: {preview}")
            
            # Exibir metadados se disponíveis
            if hasattr(source_node.node, 'metadata') and source_node.node.metadata:
                print(f"Metadata: {source_node.node.metadata}")
            
            print("-" * 50)
    else:
        print("Nenhum documento fonte encontrado na resposta.")
        print("Isso pode acontecer se o modelo estiver usando apenas seu conhecimento base.")
    print("=" * 30)

def display_similar_documents(chroma_collection, query_embeddings):
    
    distances = chroma_collection.query(
            query_embeddings=query_embeddings,
            n_results=30,
            include=["distances", "documents", "metadatas"]
    )
    
    print("\nDocumentos mais similares (via embedding):")
    if "ids" in distances and distances["ids"]:
        for i, (doc_id, distance) in enumerate(zip(distances["ids"][0], distances["distances"][0])):
            print(f"Doc ID: {doc_id}, Distância: {distance}")
            if "documents" in distances and i < len(distances["documents"][0]):
                doc_preview = distances["documents"][0][i]
                if len(doc_preview) > 150:
                    doc_preview = doc_preview[:150] + "..."
                print(f"Conteúdo: {doc_preview}")
            print("-" * 30)
            
