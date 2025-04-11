import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import { formatBotMessage } from './tools/formatting';

function App() {
  const [messages, setMessages] = useState([
    { text: "Olá! Sou seu assistente de investimentos. Como posso ajudar?", sender: "bot" }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Função para rolar automaticamente para a mensagem mais recente
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (input.trim() === '') return;

    // Adicionar mensagem do usuário ao chat
    const userMessage = { text: input, sender: "user" };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Enviar requisição para a API
      const response = await fetch('/investments/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: input }),
      });

      if (!response.ok) {
        throw new Error('Erro ao buscar resposta');
      }

      const data = await response.json();
      
      // Se a resposta vier em um objeto com campo "response"
      const responseText = typeof data === 'object' && data.response 
        ? data.response 
        : data;
      
      // Adicionar resposta do bot ao chat
      setMessages(prevMessages => [...prevMessages, { 
        text: responseText, 
        sender: "bot",
        // Se houver documentos fonte na resposta
        sources: data.source_documents || []
      }]);
    } catch (error) {
      console.error('Erro:', error);
      setMessages(prevMessages => [...prevMessages, 
        { text: "Desculpe, tive um problema para processar sua pergunta. Tente novamente mais tarde.", sender: "bot" }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Assistente de Investimentos</h1>
      </div>
      
      <div className="messages-container">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            <div className="message-bubble">
              {message.sender === 'bot' 
                ? formatBotMessage(message.text) 
                : message.text}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message bot">
            <div className="message-bubble loading">
              <div className="loading-dot"></div>
              <div className="loading-dot"></div>
              <div className="loading-dot"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Digite sua pergunta sobre investimentos..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || input.trim() === ''}>
          Enviar
        </button>
      </form>
    </div>
  );
}

export default App;