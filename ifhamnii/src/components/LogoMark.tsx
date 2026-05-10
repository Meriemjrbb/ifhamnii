import { useState } from "react"

type LogoMarkProps = {
  size?: "sm" | "md" | "lg" | "xl"
  className?: string
}

const sizes = {
  sm: "w-10 h-10 text-sm",
  md: "w-12 h-12 text-base",
  lg: "w-16 h-16 text-xl",
  xl: "w-24 h-24 text-3xl",
}

export default function LogoMark({ size = "md", className = "" }: LogoMarkProps) {
  const [loaded, setLoaded] = useState(false)

  return (
    <div
      className={`${sizes[size]} ifhamnii-logo relative shrink-0 overflow-hidden rounded-full bg-white shadow-md ${className}`}
    >
      <img
        src="/logo.png"
        alt="Ifhamnii logo"
        className={`h-full w-full rounded-full object-contain ${loaded ? "block" : "hidden"}`}
        onLoad={() => setLoaded(true)}
        onError={() => setLoaded(false)}
      />
      {!loaded && (
        <div className="flex h-full w-full items-center justify-center rounded-full bg-[#E8F5F1]">
          <span className="font-black text-[#1B5E4F]">إف</span>
        </div>
      )}
    </div>
  )
}
