import { useNavigate } from "react-router"
import { motion } from "motion/react"
import { useAuth } from "../context/AuthContext"
import { Camera, Upload, BookOpen, History, Settings } from "lucide-react"
import LogoMark from "../components/LogoMark"

export default function Home() {
  const navigate = useNavigate()
  const { profile } = useAuth()

  return (
    <div
      dir="rtl"
      className="min-h-screen bg-gradient-to-br from-[#E8F5F1] via-white to-[#F1FAEE] px-6 py-8"
    >
      <div className="mx-auto w-full max-w-5xl">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8"
        >
          {/* Utilisateur */}
          <div className="flex items-center gap-3">
            <LogoMark size="md" />
            <div>
              <h2 className="text-lg font-bold text-[#1B4332]">أهلاً بك</h2>
              <p className="text-[#52B69A] font-bold">
                {profile?.full_name || "مستخدم"}
              </p>
            </div>
          </div>

          {/* Bouton Settings */}
          <motion.button
            whileHover={{ rotate: 90 }}
            transition={{ duration: 0.3 }}
            onClick={() => navigate("/settings")}
            className="w-10 h-10 rounded-full bg-white/80 flex items-center justify-center shadow-md"
          >
            <Settings className="text-[#1B5E4F]" size={20} />
          </motion.button>
        </motion.div>

        {/* Cartes principales */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="flex flex-col gap-6 mb-8"
        >

          {/* Carte Caméra */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate("/camera")}
            className="relative h-52 rounded-[30px] overflow-hidden cursor-pointer shadow-xl"
          >
            {/* Image de fond */}
            <div
              className="absolute inset-0 bg-cover bg-center"
              style={{
                backgroundImage: `url(https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800)`
              }}
            />
            {/* Overlay vert semi-transparent par dessus l'image */}
            <div className="absolute inset-0 bg-gradient-to-br from-[#1B5E4F]/80 to-[#52796F]/80 backdrop-blur-sm" />

            {/* Contenu par dessus l'overlay */}
            <div className="relative h-full flex flex-col items-center justify-center text-center p-6">
              <div className="w-20 h-20 mb-4 rounded-full bg-white/20 flex items-center justify-center">
                <Camera className="text-white" size={40} />
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">ترجمة مباشرة</h3>
              <p className="text-white/90 text-sm">اترجم لغة الإشارة في الوقت الفعلي</p>
            </div>
          </motion.div>

          {/* Carte Upload */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate("/upload")}
            className="relative h-52 rounded-[30px] overflow-hidden cursor-pointer shadow-xl"
          >
            {/* Fond dégradé vert */}
            <div className="absolute inset-0 bg-gradient-to-br from-[#74C69D] to-[#52796F]" />

            {/* Cercles décoratifs */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16" />
            <div className="absolute bottom-0 left-0 w-40 h-40 bg-[#52B69A]/20 rounded-full -ml-20 -mb-20" />

            {/* Contenu */}
            <div className="relative h-full flex flex-col items-center justify-center text-center p-6">
              <div className="w-20 h-20 mb-4 rounded-full bg-white/20 flex items-center justify-center">
                <Upload className="text-white" size={40} />
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">رفع فيديو</h3>
              <p className="text-white/90 text-sm">ارفع فيديو لترجمته</p>
            </div>
          </motion.div>

        </motion.div>

        {/* Raccourcis en bas */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="flex justify-center gap-10"
        >

          {/* Dictionnaire */}
          <motion.div
            whileHover={{ scale: 1.1, y: -5 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate("/dictionary")}
            className="flex flex-col items-center cursor-pointer"
          >
            <div className="w-16 h-16 rounded-full bg-white shadow-lg flex items-center justify-center mb-2">
              <BookOpen className="text-[#1B5E4F]" size={28} />
            </div>
            <p className="text-sm font-medium text-[#1B4332]">قاموس الإشارات</p>
          </motion.div>

          {/* Historique */}
          <motion.div
            whileHover={{ scale: 1.1, y: -5 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate("/history")}
            className="flex flex-col items-center cursor-pointer"
          >
            <div className="w-16 h-16 rounded-full bg-white shadow-lg flex items-center justify-center mb-2">
              <History className="text-[#1B5E4F]" size={28} />
            </div>
            <p className="text-sm font-medium text-[#1B4332]">السجل</p>
          </motion.div>

        </motion.div>

      </div>
    </div>
  )
}
