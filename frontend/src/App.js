import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Homepage from './pages/homepage';

function App() {
  return (
    <Router>
      <div
        style={{
          background: "linear-gradient(135deg, #0f4834ff 0%, #409a7bff 100%)",
          minHeight: "100vh"
        }}
      >

        { /* App Routes */ }
        <Routes>
          <Route path="/" element={<Homepage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
