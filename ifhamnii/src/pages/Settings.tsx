import { useEffect, useState } from "react"
import { useNavigate } from "react-router"
import { useAuth } from "../context/AuthContext"
import { motion } from "motion/react"
import { ArrowRight, BookOpen, Moon, LogOut, ChevronLeft } from "lucide-react"
import LogoMark from "../components/LogoMark"
import { readSavedDarkMode, saveDarkMode } from "../lib/theme"

export default function Settings() {
  const navigate = useNavigate()
  const { profile, signOut } = useAuth()
  const [darkMode, setDarkMode] = useState(readSavedDarkMode)

  useEffect(() => {
    saveDarkMode(darkMode)
  }, [darkMode])

  const toggleDarkMode = () => {
    setDarkMode(current => {
      const next = !current
      saveDarkMode(next)
      return next
    })
  }

  const pageClass = darkMode
    ? "min-h-screen bg-gradient-to-br from-[#0B1F1A] via-[#102A24] to-[#071512]"
    : "min-h-screen bg-gradient-to-br from-[#E8F5F1] via-white to-[#E8F5F1]"
  const cardClass = darkMode
    ? "bg-[#132E27] rounded-3xl p-6 shadow-md flex items-center gap-4 mb-6 border border-[#21483E]"
    : "bg-white rounded-3xl p-6 shadow-md flex items-center gap-4 mb-6"
  const sectionClass = darkMode
    ? "bg-[#132E27] rounded-3xl shadow-md overflow-hidden border border-[#21483E]"
    : "bg-white rounded-3xl shadow-md overflow-hidden"
  const titleClass = darkMode ? "text-[#E8F5F1]" : "text-[#1B4332]"
  const mutedClass = darkMode ? "text-[#A9CFC4]" : "text-[#52796F]"
  const iconBgClass = darkMode ? "bg-[#1B4332]" : "bg-[#E8F5F1]"
  const dividerClass = darkMode ? "border-[#21483E]" : "border-gray-50"

  const ToggleSwitch = ({ value, onChange }: { value: boolean, onChange: () => void }) => (
    <button
      onClick={onChange}
      className={`w-12 h-6 rounded-full transition-all duration-300 relative ${value ? "bg-[#1B5E4F]" : "bg-gray-200"
        }`}
    >
      <div className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-all duration-300 ${value ? "right-1" : "left-1"
        }`} />
    </button>
  )

  return (
    <div dir="rtl" className={pageClass}>
      <main className="mx-auto min-h-screen w-full max-w-4xl px-5 py-6 sm:px-8">
        {/* Header */}
        <div className="pb-4">
          <div className="mb-6 flex items-center gap-3">
            <button
              onClick={() => navigate("/home")}
              className={`flex h-10 w-10 items-center justify-center rounded-full shadow-md ${darkMode ? "bg-[#132E27]" : "bg-white"
                }`}
            >
              <ArrowRight className="text-[#1B5E4F]" size={18} />
            </button>
            <h1 className={`text-2xl font-black ${titleClass}`}>الإعدادات</h1>
          </div>

          {/* Profil */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`${cardClass} sm:p-8`}
          >
            <LogoMark size="lg" />
            <div>
              <p className={`font-bold text-lg ${titleClass}`}>
                {profile?.full_name || "مستخدم"}
              </p>
              <p className={`text-sm ${mutedClass}`}>
                {profile?.email || ""}
              </p>
            </div>
          </motion.div>
        </div>

        <div className="flex flex-col gap-4">

          {/* Section Préférences */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className={`${sectionClass} sm:rounded-[28px]`}
          >
            <p className={`px-5 pt-4 pb-2 text-xs font-bold ${mutedClass}`}>التفضيلات</p>

            {/* Mode sombre */}
            <div className="flex items-center justify-between px-5 py-4">
              <div className="flex items-center gap-3">
                <div className={`w-9 h-9 rounded-full flex items-center justify-center ${iconBgClass}`}>
                  <Moon className="text-[#1B5E4F]" size={18} />
                </div>
                <p className={`font-medium ${titleClass}`}>الوضع الداكن</p>
              </div>
              <ToggleSwitch value={darkMode} onChange={toggleDarkMode} />
            </div>
          </motion.div>

          {/* Section Application */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className={`${sectionClass} sm:rounded-[28px]`}
          >
            <p className={`px-5 pt-4 pb-2 text-xs font-bold ${mutedClass}`}>التطبيق</p>

            {/* Tutorial */}
            <button
              onClick={() => navigate("/tutorial")}
              className={`w-full flex items-center justify-between px-5 py-4 ${dividerClass}`}
            >
              <div className="flex items-center gap-3">
                <div className={`w-9 h-9 rounded-full flex items-center justify-center ${iconBgClass}`}>
                  <BookOpen className="text-[#1B5E4F]" size={18} />
                </div>
                <p className={`font-medium ${titleClass}`}>دليل الاستخدام</p>
              </div>
              <ChevronLeft className={mutedClass} size={16} />
            </button>
          </motion.div>

          {/* Version */}
          <p className={`text-center text-xs mt-2 ${mutedClass}`}>
            إفهمني — الإصدار 1.0.0
          </p>

          {/* Déconnexion */}
          <motion.button
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            whileTap={{ scale: 0.97 }}
            onClick={async () => {
              await signOut()
              navigate("/")
            }}
            className={`flex items-center justify-center gap-3 w-full py-4 rounded-2xl font-bold mb-8 ${darkMode ? "bg-red-500/10 text-red-300" : "bg-red-50 text-red-400"
              }`}
          >
            <LogOut size={20} />
            تسجيل الخروج
          </motion.button>

        </div>
      </main>
    </div>
  )
}
