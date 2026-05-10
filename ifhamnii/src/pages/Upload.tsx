import { useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router"
import { motion } from "motion/react"
import { ArrowRight, CheckCircle, Loader, UploadCloud } from "lucide-react"
import { useAuth } from "../context/AuthContext"
import { supabase } from "../lib/supabase"

const API_URL = "http://localhost:5000/predict"

export default function Upload() {
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const { user } = useAuth()
  const [mode, setMode] = useState<"idle" | "processing" | "result">("idle")
  const [result, setResult] = useState("")
  const [error, setError] = useState("")
  const [fileName, setFileName] = useState("")

  useEffect(() => {
    inputRef.current?.click()
  }, [])

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setFileName(file.name)
    setError("")
    setMode("processing")

    try {
      const formData = new FormData()
      formData.append("video", file)

      const response = await fetch(API_URL, { method: "POST", body: formData })
      const data = await response.json()

      if (data.error) {
        setError(data.error)
        setMode("idle")
        return
      }

      setResult(data.text)
      if (user && data.text) {
        await supabase.from("history").insert({
          user_id: user.id,
          text: data.text,
          duration: null,
        })
      }
      setMode("result")
    } catch {
      setError("خطأ في الاتصال بالخادم")
      setMode("idle")
    }
  }

  const chooseAnother = () => {
    setMode("idle")
    setResult("")
    setFileName("")
    inputRef.current?.click()
  }

  return (
    <main
      dir="rtl"
      className="min-h-screen bg-gradient-to-br from-[#E8F5F1] via-white to-[#E8F5F1] px-5 py-6 sm:px-8 lg:px-12"
    >
      <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={handleFileUpload} />

      <div className="mx-auto flex min-h-[calc(100vh-48px)] w-full max-w-4xl flex-col">
        <header className="mb-6 flex items-center gap-4">
          <button
            onClick={() => navigate("/home")}
            className="flex h-11 w-11 items-center justify-center rounded-full bg-white text-[#1B5E4F] shadow-md"
          >
            <ArrowRight size={20} />
          </button>
          <div>
            <h1 className="text-2xl font-black text-[#1B4332]">رفع فيديو</h1>
            <p className="text-sm font-bold text-[#52796F]">اختر فيديو واضح لترجمته</p>
          </div>
        </header>

        <section className="flex flex-1 items-center justify-center">
          {mode === "idle" && (
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full rounded-[32px] bg-white/90 p-6 text-center shadow-xl ring-1 ring-[#D8F3DC] sm:p-10"
            >
              <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-3xl bg-[#E8F5F1] text-[#1B5E4F]">
                <UploadCloud size={38} />
              </div>
              <h2 className="text-3xl font-black text-[#1B4332]">اختر فيديو</h2>
              <p className="mx-auto mt-3 max-w-md text-sm leading-6 text-[#52796F] sm:text-base">
                يفضل أن يكون الفيديو قصيرا، بإضاءة واضحة، وتظهر اليدان فيه بشكل كامل.
              </p>

              {error && (
                <div className="mx-auto mt-5 max-w-md rounded-2xl border border-red-200 bg-red-50 p-4 text-sm font-bold text-red-500">
                  {error}
                </div>
              )}

              <button
                onClick={() => inputRef.current?.click()}
                className="mx-auto mt-8 w-full max-w-sm rounded-full bg-gradient-to-r from-[#52B69A] to-[#1B5E4F] px-8 py-4 text-lg font-black text-white shadow-lg"
              >
                اختيار فيديو
              </button>
            </motion.div>
          )}

          {mode === "processing" && (
            <div className="flex flex-col items-center text-center">
              <Loader className="animate-spin text-[#1B5E4F]" size={58} />
              <p className="mt-6 text-2xl font-black text-[#1B4332]">جارٍ التحليل...</p>
              <p className="mt-2 max-w-md truncate text-sm font-bold text-[#52796F]">{fileName}</p>
            </div>
          )}

          {mode === "result" && (
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full max-w-2xl"
            >
              <div className="rounded-[32px] bg-white p-6 text-center shadow-xl ring-1 ring-[#D8F3DC] sm:p-9">
                <CheckCircle className="mx-auto text-[#52B69A]" size={62} />
                <p className="mt-5 text-sm font-bold text-[#52796F]">النص المترجم</p>
                <p className="mt-3 text-2xl font-black leading-10 text-[#1B4332]">{result}</p>
              </div>

              <div className="mt-5 grid gap-3 sm:grid-cols-2">
                <button
                  onClick={chooseAnother}
                  className="rounded-full bg-gradient-to-r from-[#52B69A] to-[#1B5E4F] px-8 py-4 font-black text-white shadow-lg"
                >
                  رفع فيديو جديد
                </button>
                <button
                  onClick={() => navigate("/history")}
                  className="rounded-full bg-white px-8 py-4 font-black text-[#1B5E4F] shadow-md ring-1 ring-[#52B69A]"
                >
                  فتح السجل
                </button>
              </div>
            </motion.div>
          )}
        </section>
      </div>
    </main>
  )
}
