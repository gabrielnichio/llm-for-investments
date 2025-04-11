export const formatBotMessage = (text) => {
    if (!text) return '';
    
    // Divide o texto por quebras de linha e mapeia para elementos JSX
    return text.split('\n').map((line, index) => {
      // Verifica se a linha é um item numerado (como "1. SOJA3: R$ 6,92")
      const isListItem = /^\d+\.\s/.test(line);
      
      // Se for um item de lista, aplica estilo de lista
      if (isListItem) {
        return (
          <div key={index} className="list-item">
            <span>{line}</span>
          </div>
        );
      }
      
      // Se for uma linha em branco, adiciona espaço
      if (line.trim() === '') {
        return <div key={index} className="empty-line" />;
      }
      
      // Verifica se é um total ou conclusão
      const isTotal = line.toLowerCase().includes('total') || line.startsWith('Estes são');
      
      // Retorna um parágrafo normal
      return (
        <div key={index} className={isTotal ? "highlight" : "text-line"}>
          {line}
        </div>
      );
    });
};