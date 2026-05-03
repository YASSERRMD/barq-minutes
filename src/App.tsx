import { NavLink, Route, Routes } from 'react-router-dom';
import { FileText, Mic, Settings as SettingsIcon } from 'lucide-react';
import Dashboard from './routes/Dashboard';
import Meeting from './routes/Meeting';
import Recording from './routes/Recording';
import Settings from './routes/Settings';
import { ModelBootProvider } from './context/ModelBootContext';

function navClass({ isActive }: { isActive: boolean }) {
  return isActive ? 'nav-link active' : 'nav-link';
}

function AppShell() {
  return (
    <div className="app-frame">
      <aside className="sidebar" aria-label="Primary">
        <NavLink to="/" className="brand" aria-label="barq-minutes dashboard">
          <span className="brand-mark" />
          <span>barq-minutes</span>
        </NavLink>
        <nav className="nav-list">
          <NavLink to="/" className={navClass} end>
            <FileText size={18} />
            Meetings
          </NavLink>
          <NavLink to="/record" className={navClass}>
            <Mic size={18} />
            Record
          </NavLink>
          <NavLink to="/settings" className={navClass}>
            <SettingsIcon size={18} />
            Settings
          </NavLink>
        </nav>
      </aside>
      <main className="page-shell">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/record" element={<Recording />} />
          <Route path="/meeting/:id" element={<Meeting />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ModelBootProvider>
      <AppShell />
    </ModelBootProvider>
  );
}
