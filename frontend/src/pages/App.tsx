import { useState } from "react";
import { BookOpen, Bot, Database, Image, ListChecks, MessageSquare, Settings as SettingsIcon, Users } from "lucide-react";
import { ChatPage } from "./ChatPage";
import { JobsPage } from "./JobsPage";
import { LibraryPage } from "./LibraryPage";
import { ModelsPage } from "./ModelsPage";
import { PhotosPage } from "./PhotosPage";
import { PeoplePage } from "./PeoplePage";
import { SettingsPage } from "./SettingsPage";

type Page = "chat" | "library" | "photos" | "people" | "jobs" | "models" | "settings";

const nav = [
  { key: "chat" as const, label: "Chat", icon: MessageSquare },
  { key: "library" as const, label: "Library", icon: Database },
  { key: "photos" as const, label: "Photos", icon: Image },
  { key: "people" as const, label: "People", icon: Users },
  { key: "jobs" as const, label: "Jobs", icon: ListChecks },
  { key: "models" as const, label: "Models", icon: Bot },
  { key: "settings" as const, label: "Settings", icon: SettingsIcon }
];

export function App() {
  const [page, setPage] = useState<Page>("chat");
  const ActivePage = {
    chat: ChatPage,
    library: LibraryPage,
    photos: PhotosPage,
    people: PeoplePage,
    jobs: JobsPage,
    models: ModelsPage,
    settings: SettingsPage
  }[page];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <BookOpen size={22} />
          <span>Lentica</span>
        </div>
        <nav>
          {nav.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.key}
                className={page === item.key ? "nav-item active" : "nav-item"}
                onClick={() => setPage(item.key)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </aside>
      <main className="main-panel">
        <ActivePage />
      </main>
    </div>
  );
}
