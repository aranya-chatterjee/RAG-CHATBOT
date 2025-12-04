export function TeacherDoodle() {
  return (
    <svg
      width="60"
      height="60"
      viewBox="0 0 60 60"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="flex-shrink-0"
    >
      {/* Head */}
      <circle cx="30" cy="18" r="8" fill="#f4a460" stroke="#d4860a" strokeWidth="1.5" />

      {/* Turban/Head wrap */}
      <path d="M 22 16 Q 22 10 30 10 Q 38 10 38 16" fill="#e67e22" stroke="#d4860a" strokeWidth="1.5" />
      <path d="M 22 16 Q 22 12 30 12 Q 38 12 38 16" fill="#f39c12" stroke="#d4860a" strokeWidth="1" />

      {/* Eyes */}
      <circle cx="27" cy="17" r="1.5" fill="#2c3e50" />
      <circle cx="33" cy="17" r="1.5" fill="#2c3e50" />

      {/* Smile */}
      <path d="M 27 20 Q 30 21.5 33 20" stroke="#2c3e50" strokeWidth="1.5" fill="none" strokeLinecap="round" />

      {/* Mustache */}
      <path d="M 30 19 Q 27 19.5 25 19" stroke="#8b4513" strokeWidth="1.5" fill="none" strokeLinecap="round" />
      <path d="M 30 19 Q 33 19.5 35 19" stroke="#8b4513" strokeWidth="1.5" fill="none" strokeLinecap="round" />

      {/* Body - Traditional Kurta */}
      <path
        d="M 22 26 L 20 42 Q 20 44 22 44 L 38 44 Q 40 44 40 42 L 38 26 Z"
        fill="#c0392b"
        stroke="#a93226"
        strokeWidth="1.5"
      />

      {/* Kurta pattern */}
      <line x1="30" y1="26" x2="30" y2="44" stroke="#a93226" strokeWidth="1" opacity="0.5" />
      <circle cx="30" cy="32" r="1.5" fill="#f39c12" />
      <circle cx="30" cy="38" r="1.5" fill="#f39c12" />

      {/* Arms */}
      <line x1="22" y1="28" x2="12" y2="32" stroke="#f4a460" strokeWidth="2.5" strokeLinecap="round" />
      <line x1="38" y1="28" x2="48" y2="32" stroke="#f4a460" strokeWidth="2.5" strokeLinecap="round" />

      {/* Hands */}
      <circle cx="12" cy="32" r="2" fill="#f4a460" stroke="#d4860a" strokeWidth="1" />
      <circle cx="48" cy="32" r="2" fill="#f4a460" stroke="#d4860a" strokeWidth="1" />

      {/* Legs */}
      <line x1="25" y1="44" x2="25" y2="54" stroke="#2c3e50" strokeWidth="2" strokeLinecap="round" />
      <line x1="35" y1="44" x2="35" y2="54" stroke="#2c3e50" strokeWidth="2" strokeLinecap="round" />

      {/* Feet */}
      <ellipse cx="25" cy="55" rx="2.5" ry="1.5" fill="#2c3e50" />
      <ellipse cx="35" cy="55" rx="2.5" ry="1.5" fill="#2c3e50" />

      {/* Pointer stick */}
      <line x1="48" y1="32" x2="52" y2="20" stroke="#8b4513" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="52" cy="19" r="1.5" fill="#f39c12" />
    </svg>
  )
}
