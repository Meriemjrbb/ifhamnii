import { useState } from "react"
import { useNavigate } from "react-router"
import { motion } from "motion/react"
import { Eye, EyeOff, Mail, Lock } from "lucide-react"
import { supabase } from "../lib/supabase"
import LogoMark from "../components/LogoMark"

export default function Login() {
  const navigate = useNavigate()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const handleLogin = async () => {
    if (!email || !password) {
      setError("يرجى إدخال البريد الإلكتروني وكلمة المرور")
      return
    }

    setLoading(true)
    setError("")

    const { error } = await supabase.auth.signInWithPassword({
      email,
      password
    })

    setLoading(false)

    if (error) {
      if (error.message.includes("Email not confirmed")) {
        setError("يرجى تأكيد بريدك الإلكتروني أولاً — تحقق من صندوق الوارد")
      } else if (error.message.includes("Invalid login")) {
        setError("البريد الإلكتروني أو كلمة المرور غير صحيحة")
      } else {
        setError(error.message)
      }
    } else {
      navigate("/home")
    }
  }

  return (
    <div
      dir="rtl"
      className="min-h-screen bg-gradient-to-br from-[#E8F5F1] via-white to-[#E8F5F1] flex flex-col items-center justify-center px-6"
    >
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-md"
      >
        <div className="text-center mb-10">
          <LogoMark size="xl" className="mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-[#1B4332]">تسجيل الدخول</h2>
          <p className="text-[#52796F] mt-2">مرحباً بعودتك</p>
        </div>

        <div className="bg-white rounded-3xl shadow-xl p-8 flex flex-col gap-5">

          {/* Email */}
          <div className="flex flex-col gap-2">
            <label className="text-[#1B4332] font-medium">البريد الإلكتروني</label>
            <div className="flex items-center gap-3 bg-[#F1FAEE] rounded-2xl px-4 py-3">
              <Mail className="text-[#52B69A]" size={20} />
              <input
                type="email"
                placeholder="example@email.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="bg-transparent flex-1 outline-none text-[#1B4332] placeholder:text-[#52796F]/50 text-right"
              />
            </div>
          </div>

          {/* Mot de passe */}
          <div className="flex flex-col gap-2">
            <label className="text-[#1B4332] font-medium">كلمة المرور</label>
            <div className="flex items-center gap-3 bg-[#F1FAEE] rounded-2xl px-4 py-3">
              <button onClick={() => setShowPassword(!showPassword)}>
                {showPassword
                  ? <EyeOff className="text-[#52B69A]" size={20} />
                  : <Eye className="text-[#52B69A]" size={20} />
                }
              </button>
              <input
                type={showPassword ? "text" : "password"}
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="bg-transparent flex-1 outline-none text-[#1B4332] placeholder:text-[#52796F]/50 text-right"
              />
              <Lock className="text-[#52B69A]" size={20} />
            </div>
          </div>

          {/* Erreur */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-2xl p-3">
              <p className="text-red-500 text-sm text-center">{error}</p>
            </div>
          )}

          {/* Bouton */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={handleLogin}
            disabled={loading}
            className="w-full py-4 bg-gradient-to-r from-[#52B69A] to-[#1B5E4F] text-white rounded-full font-bold text-lg shadow-lg mt-2 disabled:opacity-50"
          >
            {loading ? "جارٍ الدخول..." : "دخول"}
          </motion.button>

          {/* Lien Register */}
          <p className="text-center text-[#52796F]">
            ليس لديك حساب؟{" "}
            <button
              onClick={() => navigate("/register")}
              className="text-[#1B5E4F] font-bold"
            >
              إنشاء حساب
            </button>
          </p>

        </div>
      </motion.div>
    </div>
  )
}
