import './Message.css'



function Message({ sender, text }) {
  return (
    <div className={`message ${sender === 'user' ? 'user-message' : 'bot-message'}`}>
      <p>{text}</p>
    </div>
  )
}

export default Message