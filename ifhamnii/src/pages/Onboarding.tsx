import { useNavigate } from "react-router"
import { motion } from "motion/react"
import LogoMark from "../components/LogoMark"

const heroImage = "https://images.unsplash.com/photo-1655720362153-1bbfa72c2d13?w=1200"
const heroVideo = "/onboarding-video.mp4"

export default function Onboarding() {
  const navigate = useNavigate()

  return (
    <main
      dir="rtl"
      className="min-h-screen bg-gradient-to-br from-[#E8F5F1] via-white to-[#E8F5F1] px-5 py-5 sm:px-8 lg:px-12"
    >
      <div className="mx-auto flex min-h-[calc(100vh-40px)] w-full max-w-6xl flex-col justify-center">
        <motion.section
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55 }}
          className="grid items-center gap-8 lg:grid-cols-[1.05fr_0.95fr] lg:gap-12"
        >
          <div className="relative order-1 lg:order-2">
            <div className="relative aspect-video overflow-hidden rounded-[34px] bg-[#081C15] shadow-2xl">
              <video
                src={heroVideo}
                poster={heroImage}
                autoPlay
                muted
                loop
                playsInline
                className="h-full w-full object-contain"
              >
                <img src={heroImage} alt="لغة الإشارة" />
              </video>
              <div className="absolute inset-0 rounded-[34px] bg-gradient-to-t from-[#081C15]/55 via-transparent to-transparent" />
            </div>

            <div className="absolute bottom-5 right-5 rounded-full bg-white/90 px-4 py-2 text-sm font-bold text-[#1B5E4F] shadow-lg backdrop-blur">
              لغة الإشارة العربية
            </div>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, delay: 0.15 }}
            className="order-2 flex flex-col items-center text-center lg:order-1"
          >
            <LogoMark size="xl" className="mb-6" />

            <h1 className="max-w-xl text-4xl font-black leading-tight text-[#1B4332] sm:text-5xl lg:text-6xl">
              جسرك للتواصل مع الجميع
            </h1>

            <p className="mt-5 max-w-lg text-base leading-8 text-[#52796F] sm:text-lg">
              ترجمة لغة الإشارة العربية إلى نص مكتوب بطريقة بسيطة، واضحة، وسريعة.
            </p>

            <div className="mt-7 grid w-full max-w-md grid-cols-3 gap-2 text-center text-xs font-bold text-[#1B5E4F] sm:text-sm">
              <div className="rounded-2xl bg-white/80 px-3 py-3 shadow-sm ring-1 ring-[#D8F3DC]">
                كاميرا
              </div>
              <div className="rounded-2xl bg-white/80 px-3 py-3 shadow-sm ring-1 ring-[#D8F3DC]">
                فيديو
              </div>
              <div className="rounded-2xl bg-white/80 px-3 py-3 shadow-sm ring-1 ring-[#D8F3DC]">
                قاموس
              </div>
            </div>

            <div className="mt-9 flex w-full max-w-md flex-col gap-3">
              <motion.button
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.45, delay: 0.35 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => navigate("/login")}
                className="w-full rounded-full bg-gradient-to-r from-[#52B69A] to-[#1B5E4F] px-8 py-4 text-lg font-black text-white shadow-xl"
              >
                ابدأ الآن
              </motion.button>

              <p className="text-center text-sm text-[#52796F] sm:text-base">
                لديك حساب بالفعل؟{" "}
                <button
                  onClick={() => navigate("/login")}
                  className="font-black text-[#1B5E4F]"
                >
                  سجل الدخول
                </button>
              </p>
            </div>
          </motion.div>
        </motion.section>
      </div>
    </main>
  )
}
