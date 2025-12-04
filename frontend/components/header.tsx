import { TeacherDoodle } from "./teacher-doodle"

export function Header() {
  return (
    <header className="border-b border-border bg-card px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <TeacherDoodle />
          <div>
            <h1
              className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-orange-400"
              style={{ fontFamily: "'Playfair Display', serif" }}
            >
              <span className="text-amber-400">M</span>asterji
            </h1>
            <p className="text-sm text-muted-foreground">Your AI Tutor - Learn Smarter</p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span>Ready to help</span>
        </div>
      </div>
    </header>
  )
}
