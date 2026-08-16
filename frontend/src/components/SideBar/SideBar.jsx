import React, { useState } from "react";
import "./Sidebar.css";
import HazardEx from '../../assets/HazardEx.png'


const NAV_ITEMS = [
  {
    id: "chat",
    label: "Chat History",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
      </svg>
    ),
  },
  {
    id: "knowledge",
    label: "Knowledge Base",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
      </svg>
    ),
  },
];

export default function Sidebar({ active = "chat", onSelect }) {
  const [current, setCurrent] = useState(active);

  const handleClick = (id) => {
    setCurrent(id);
    if (onSelect) onSelect(id);
  };

  return (
    <aside className="hx-sidebar">
      <div>
        <img src={HazardEx} alt="HazardEx Logo" className="hx-logo-image" />
      </div>

      <nav className="hx-sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`hx-nav-item ${current === item.id ? "active" : ""}`}
            onClick={() => handleClick(item.id)}
          >
            <span className="hx-nav-icon">{item.icon}</span>
            <span className="hx-nav-label">{item.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
}