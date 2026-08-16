import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import Chat from './components/Chat/Chat.jsx'
import Sidebar from './components/SideBar/SideBar.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <div className="hx-app">
      <Sidebar />
      <Chat />
    </div>
  </StrictMode>,
)