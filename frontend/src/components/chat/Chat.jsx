import './Chat.css'



function Chat() {
  return (
    <div className="chat-container">
        <div className="chat-header">
            <h2>HazardEx Chat</h2>
        </div>
        <div className="chat-messages">
            <div className="message user-message">
                <p>Hello! How can I assist you today?</p>
            </div>
            <div className="message bot-message">
                <p>Hi! I'm here to help you with any questions you have.</p>
            </div>
        </div>
        <div className="chat-input">
            <input type="text" placeholder="Type your message..." />
            <button>Send</button>
        </div>
    </div>
)
}
export default Chat