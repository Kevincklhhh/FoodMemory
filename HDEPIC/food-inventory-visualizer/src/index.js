import React from 'react';
import ReactDOM from 'react-dom/client';
// Switch between Apps:
// import App from './App';  // Original Food Graph Visualizer
import App from './InventoryApp';  // Inventory Lifecycle Visualizer

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
