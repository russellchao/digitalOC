import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Homepage from './pages/homepage';

function App() {
  return (
    <Router>
      <div 
        style={{ 
          paddingTop: "60px", 
          background: "linear-gradient(135deg, #0f2027 0%, #2c5364 100%)",
          minHeight: "100vh"
        }}>

        { /* App Routes */ }
        <Routes>
          <Route path="/" element={<Homepage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
