import { useRef, useState, useEffect } from "react"
import { useNavigate } from "react-router"
import { supabase } from "../lib/supabase"
import { useAuth } from "../context/AuthContext"
import { motion } from "motion/react"
import { ArrowRight, Loader, CheckCircle } from "lucide-react"

const API_URL = "http://localhost:5000/predict"

export default function Camera() {
  const navigate = useNavigate()
  const { user } = useAuth()

  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const [mode, setMode] = useState<"starting" | "recording" | "processing" | "result">("starting")
  const [result, setResult] = useState("")
  const [error, setError] = useState("")
  const [stream, setStream] = useState<MediaStream | null>(null)

  // ── Lance la caméra automatiquement dès que la page s'ouvre ──
  useEffect(() => {
    startCamera()
    // nettoie le stream quand on quitte la page
    return () => {
      stream?.getTracks().forEach(track => track.stop())
    }
  }, [])

  const startCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false
      })
      setStream(s)
      if (videoRef.current) {
        videoRef.current.srcObject = s
      }
      setMode("recording")
      startRecording(s)
    } catch (err) {
      setError("لا يمكن الوصول إلى الكاميرا — تحقق من الإذن")
    }
  }

  const startRecording = (s: MediaStream) => {
    chunksRef.current = []
    const recorder = new MediaRecorder(s, { mimeType: "video/webm" })
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }
    recorder.onstop = () => sendToAPI()
    mediaRecorderRef.current = recorder
    recorder.start()
  }

  const stopRecording = () => {
    mediaRecorderRef.current?.stop()
    stream?.getTracks().forEach(track => track.stop())
    setStream(null)
    setMode("processing")
  }

  const sendToAPI = async () => {
    try {
      const blob = new Blob(chunksRef.current, { type: "video/webm" })
      const formData = new FormData()
      formData.append("video", blob, "recording.webm")

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData
      })
      const data = await response.json()

      if (data.error) {
        setError(data.error)
        setMode("recording")
      } else {
        setResult(data.text)
        // sauvegarde dans la base
        if (user && data.text) {
          await supabase.from('history').insert({
            user_id: user.id,
            text: data.text,
            duration: null
          })
        }
        setMode("result")
      }
    } catch (err) {
      setError("خطأ في الاتصال بالخادم")
      setMode("recording")
    }
  }

  const reset = () => {
    setResult("")
    setError("")
    startCamera()
  }

  return (
    <div
      dir="rtl"
      className="min-h-screen bg-black flex flex-col"
    >
      {/* Header par dessus la caméra */}
      <div className="absolute top-0 right-0 left-0 z-10 flex items-center justify-between px-6 py-4">
        <motion.button
          whileTap={{ scale: 0.95 }}
          onClick={() => {
            stream?.getTracks().forEach(track => track.stop())
            navigate("/home")
          }}
          className="w-10 h-10 rounded-full bg-black/40 backdrop-blur-sm flex items-center justify-center"
        >
          <ArrowRight className="text-white" size={20} />
        </motion.button>
        <p className="text-white font-bold text-lg">ترجمة مباشرة</p>
        <div className="w-10" />
      </div>

      {/* ── STARTING — chargement caméra ── */}
      {mode === "starting" && (
        <div className="flex-1 flex flex-col items-center justify-center gap-4">
          <Loader className="text-white animate-spin" size={40} />
          <p className="text-white">جارٍ تشغيل الكاميرا...</p>
        </div>
      )}

      {/* ── RECORDING — caméra en plein écran ── */}
      {mode === "recording" && (
        <div className="flex-1 relative">
          {/* Vidéo plein écran */}
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-screen object-cover"
          />

          {/* Overlay bas */}
          <div className="absolute bottom-0 right-0 left-0 bg-gradient-to-t from-black/70 to-transparent p-8 flex flex-col items-center gap-4">

            {error && (
              <p className="text-red-400 text-sm text-center">{error}</p>
            )}

            {/* Indicateur enregistrement */}
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
              <p className="text-white text-sm">جارٍ التسجيل</p>
            </div>

            {/* Bouton Stop */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={stopRecording}
              className="w-20 h-20 rounded-full bg-white flex items-center justify-center shadow-2xl"
            >
              <div className="w-8 h-8 bg-red-500 rounded-sm" />
            </motion.button>

            <p className="text-white/70 text-sm">اضغط للإيقاف والترجمة</p>
          </div>
        </div>
      )}

      {/* ── PROCESSING ── */}
      {mode === "processing" && (
        <div className="flex-1 flex flex-col items-center justify-center gap-6 bg-gradient-to-br from-[#E8F5F1] via-white to-[#E8F5F1]">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          >
            <Loader className="text-[#1B5E4F]" size={60} />
          </motion.div>
          <div className="text-center">
            <p className="text-xl font-bold text-[#1B4332]">جارٍ التحليل...</p>
            <p className="text-[#52796F] mt-2">يتم استخراج الإشارات وترجمتها</p>
          </div>
        </div>
      )}

      {/* ── RESULT ── */}
      {mode === "result" && (
        <div
          dir="rtl"
          className="flex-1 flex flex-col items-center justify-center px-6 gap-6 bg-gradient-to-br from-[#E8F5F1] via-white to-[#E8F5F1]"
        >
          <CheckCircle className="text-[#52B69A]" size={60} />

          <div className="w-full max-w-md bg-white rounded-[30px] shadow-xl p-8">
            <p className="text-sm text-[#52796F] mb-3 text-center">النص المترجم</p>
            <p className="text-2xl font-bold text-[#1B4332] text-center leading-relaxed">
              {result}
            </p>
          </div>

          <div className="w-full max-w-md flex flex-col gap-3">
            <motion.button
              whileTap={{ scale: 0.97 }}
              onClick={reset}
              className="w-full py-4 bg-gradient-to-r from-[#52B69A] to-[#1B5E4F] text-white rounded-full font-bold text-lg shadow-lg"
            >
              تسجيل جديد
            </motion.button>
            <motion.button
              whileTap={{ scale: 0.97 }}
              onClick={() => navigate("/history")}
              className="w-full py-4 bg-white text-[#1B5E4F] rounded-full font-bold text-lg shadow-md border border-[#52B69A]"
            >
              حفظ في السجل
            </motion.button>
          </div>
        </div>
      )}
    </div>
  )
}