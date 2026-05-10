import { useState } from "react"
import { useNavigate } from "react-router"
import { motion } from "motion/react"
import { Eye, EyeOff, Mail, Lock, User } from "lucide-react"
import { supabase } from "../lib/supabase"
import LogoMark from "../components/LogoMark"

export default function Register() {
  const navigate = useNavigate()
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)
  const [emailSent, setEmailSent] = useState(false)

  const handleRegister = async () => {
    // Validation
    if (!name) {
      setError("يرجى إدخال اسمك الكامل")
      return
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      setError("البريد الإلكتروني غير صالح")
      return
    }
    if (password.length < 6) {
      setError("كلمة المرور يجب أن تكون 6 أحرف على الأقل")
      return
    }
    if (password !== confirmPassword) {
      setError("كلمتا المرور غير متطابقتين")
      return
    }

    setLoading(true)
    setError("")

    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: { full_name: name },
        emailRedirectTo: "http://localhost:5173/home"
      }
    })

    setLoading(false)

    if (error) {
      if (error.message.includes("already registered")) {
        setError("هذا البريد الإلكتروني مستخدم بالفعل")
      } else {
        setError(error.message)
      }
    } else {
      setEmailSent(true)
    }
  }

  // ── écran confirmation email ─────────────────────────────
  if (emailSent) {
    return (
      <div
        dir="rtl"
        className="min-h-screen bg-gradient-to-br from-[#E8F5F1] via-white to-[#E8F5F1] flex flex-col items-center justify-center px-6"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-md bg-white rounded-3xl shadow-xl p-8 flex flex-col items-center gap-6"
        >
          <div className="w-24 h-24 rounded-full bg-[#E8F5F1] flex items-center justify-center">
            <span className="text-5xl">📧</span>
          </div>

          <h2 className="text-2xl font-bold text-[#1B4332] text-center">
            تحقق من بريدك الإلكتروني
          </h2>

          <p className="text-[#52796F] text-center leading-relaxed">
            لقد أرسلنا رابط التأكيد إلى
            <span className="font-bold text-[#1B5E4F] block mt-1">{email}</span>
          </p>

          <div className="bg-[#E8F5F1] rounded-2xl p-4 w-full">
            <p className="text-[#52796F] text-sm text-center leading-relaxed">
              افتح بريدك الإلكتروني وانقر على رابط التأكيد ثم عد لتسجيل الدخول
            </p>
          </div>

          <motion.button
            whileTap={{ scale: 0.97 }}
            onClick={() => navigate("/login")}
            className="w-full py-4 bg-gradient-to-r from-[#52B69A] to-[#1B5E4F] text-white rounded-full font-bold text-lg shadow-lg"
          >
            الذهاب لتسجيل الدخول
          </motion.button>

          <button
            onClick={() => setEmailSent(false)}
            className="text-[#52796F] text-sm"
          >
            تغيير البريد الإلكتروني
          </button>
        </motion.div>
      </div>
    )
  }

  // ── formulaire inscription ───────────────────────────────
  return (
    <div
      dir="rtl"
      className="min-h-screen bg-gradient-to-br from-[#E8F5F1] via-white to-[#E8F5F1] flex flex-col items-center justify-center px-6 py-10"
    >
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-md"
      >
        <div className="text-center mb-8">
          <LogoMark size="xl" className="mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-[#1B4332]">إنشاء حساب جديد</h2>
          <p className="text-[#52796F] mt-2">انضم إلينا اليوم</p>
        </div>

        <div className="bg-white rounded-3xl shadow-xl p-8 flex flex-col gap-5">

          {/* Nom */}
          <div className="flex flex-col gap-2">
            <label className="text-[#1B4332] font-medium">الاسم الكامل</label>
            <div className="flex items-center gap-3 bg-[#F1FAEE] rounded-2xl px-4 py-3">
              <User className="text-[#52B69A]" size={20} />
              <input
                type="text"
                placeholder="أدخل اسمك الكامل"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="bg-transparent flex-1 outline-none text-[#1B4332] placeholder:text-[#52796F]/50 text-right"
              />
            </div>
          </div>

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

          {/* Confirmation */}
          <div className="flex flex-col gap-2">
            <label className="text-[#1B4332] font-medium">تأكيد كلمة المرور</label>
            <div className="flex items-center gap-3 bg-[#F1FAEE] rounded-2xl px-4 py-3">
              <Lock className="text-[#52B69A]" size={20} />
              <input
                type={showPassword ? "text" : "password"}
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="bg-transparent flex-1 outline-none text-[#1B4332] placeholder:text-[#52796F]/50 text-right"
              />
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
            onClick={handleRegister}
            disabled={loading}
            className="w-full py-4 bg-gradient-to-r from-[#52B69A] to-[#1B5E4F] text-white rounded-full font-bold text-lg shadow-lg mt-2 disabled:opacity-50"
          >
            {loading ? "جارٍ الإنشاء..." : "إنشاء حساب"}
          </motion.button>

          {/* Lien Login */}
          <p className="text-center text-[#52796F]">
            لديك حساب بالفعل؟{" "}
            <button
              onClick={() => navigate("/login")}
              className="text-[#1B5E4F] font-bold"
            >
              تسجيل الدخول
            </button>
          </p>

        </div>
      </motion.div>
    </div>
  )
}
