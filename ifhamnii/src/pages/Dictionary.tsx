import { useEffect, useState } from "react"
import { useNavigate } from "react-router"
import { motion } from "motion/react"
import { ArrowRight, Search, Loader, X } from "lucide-react"
import { supabase } from "../lib/supabase"

const CATEGORIES = [
  "الكل",
  "ألوان",
  "ارقام",
  "المهن",
  "العواطف و المشاعر",
  "التعليم والتعلم",
  "الإختبارات والشهادات",
  "السمات الشخصية",
  "لغات مختلفة",
  "موسيقى",
]

type Sign = {
  id: string
  name: string
  category: string
  video_path: string
}

export default function Dictionary() {
  const navigate = useNavigate()
  const [signs, setSigns] = useState<Sign[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState("")
  const [activeCategory, setActiveCategory] = useState("الكل")
  const [selectedSign, setSelectedSign] = useState<Sign | null>(null)

  // ── Charger depuis Supabase ──────────────────────────────
  useEffect(() => {
    supabase
      .from('signs')
      .select('*')
      .order('name')
      .then(({ data, error }) => {
        if (error) console.error(error)
        else setSigns(data || [])
        setLoading(false)
      })
  }, [])

  // ── Filtrage ─────────────────────────────────────────────
  const filtered = signs.filter(sign => {
    const matchCategory = activeCategory === "الكل" || sign.category === activeCategory
    const matchSearch = sign.name.includes(search)
    return matchCategory && matchSearch
  })

  return (
    <div dir="rtl" className="min-h-screen bg-gradient-to-br from-[#E8F5F1] via-white to-[#E8F5F1]">
      <div className="mx-auto w-full max-w-6xl">

        {/* Header sticky */}
        <div className="sticky top-0 z-10 bg-white/90 backdrop-blur-sm px-4 pt-6 pb-3 shadow-sm sm:px-6 lg:rounded-b-[28px]">

          <div className="flex items-center gap-3 mb-3">
            <button
              onClick={() => navigate("/home")}
              className="w-9 h-9 rounded-full bg-[#E8F5F1] flex items-center justify-center flex-shrink-0"
            >
              <ArrowRight className="text-[#1B5E4F]" size={18} />
            </button>
            <h1 className="text-2xl font-black text-[#1B4332]">قاموس الإشارات</h1>
          </div>

          <div className="mb-3 flex items-center gap-2 rounded-2xl bg-[#F1FAEE] px-4 py-3">
            <Search className="text-[#52B69A] flex-shrink-0" size={18} />
            <input
              type="text"
              placeholder="ابحث عن إشارة..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-transparent flex-1 outline-none text-[#1B4332] placeholder:text-[#52796F]/50 text-right text-sm"
            />
          </div>

          <div className="flex gap-2 overflow-x-auto pb-1">
            {CATEGORIES.map(cat => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                className={`flex-shrink-0 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${activeCategory === cat
                    ? "bg-[#1B5E4F] text-white"
                    : "bg-[#E8F5F1] text-[#52796F]"
                  }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Contenu */}
        <div className="px-4 py-5 sm:px-6">

          {/* Chargement */}
          {loading && (
            <div className="flex justify-center py-20">
              <Loader className="text-[#52B69A] animate-spin" size={40} />
            </div>
          )}

          {/* Résultats */}
          {!loading && (
            <>
              <p className="text-xs text-[#52796F] mb-3">{filtered.length} إشارة</p>

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                {filtered.map((sign, index) => (
                  <motion.div
                    key={sign.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.02 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={() => setSelectedSign(sign)}
                    className="group cursor-pointer overflow-hidden rounded-2xl bg-white shadow-md"
                  >
                    {/* Vidéo thumbnail */}
                    <div className="relative w-full bg-[#E8F5F1]" style={{ paddingTop: "75%" }}>
                      <video
                        src={sign.video_path}
                        muted
                        loop
                        playsInline
                        preload="metadata"
                        className="absolute inset-0 w-full h-full object-cover"
                        onMouseEnter={(e) => (e.target as HTMLVideoElement).play()}
                        onMouseLeave={(e) => {
                          const v = e.target as HTMLVideoElement
                          v.pause()
                          v.currentTime = 0
                        }}
                      />
                      {/* Icône play */}
                      <div className="absolute inset-0 flex items-center justify-center group-hover:opacity-0 transition-opacity">
                        <div className="w-10 h-10 rounded-full bg-[#1B5E4F]/70 flex items-center justify-center">
                          <div className="w-0 h-0 border-t-[7px] border-t-transparent border-b-[7px] border-b-transparent border-l-[12px] border-l-white ml-1" />
                        </div>
                      </div>
                    </div>

                    {/* Nom */}
                    <div className="px-3 py-2">
                      <p className="text-sm font-bold text-[#1B4332] text-center truncate">
                        {sign.name}
                      </p>
                      <p className="text-xs text-[#52796F] text-center truncate">
                        {sign.category}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>

              {filtered.length === 0 && (
                <div className="flex flex-col items-center justify-center py-20 gap-3">
                  <p className="text-4xl">🔍</p>
                  <p className="text-[#52796F] text-center text-sm">
                    لا توجد إشارة بهذا الاسم
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Modal vidéo */}
        {selectedSign && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 px-4"
            onClick={() => setSelectedSign(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-white rounded-3xl overflow-hidden w-full max-w-sm shadow-2xl"
            >
              <div className="relative bg-black">
                <video
                  key={selectedSign.video_path}
                  src={selectedSign.video_path}
                  autoPlay
                  loop
                  controls
                  playsInline
                  className="w-full"
                  style={{ maxHeight: "60vh" }}
                />
                <button
                  onClick={() => setSelectedSign(null)}
                  className="absolute top-3 left-3 w-8 h-8 rounded-full bg-black/50 flex items-center justify-center"
                >
                  <X className="text-white" size={16} />
                </button>
              </div>

              <div className="p-5 flex flex-col items-center gap-2">
                <h2 className="text-2xl font-bold text-[#1B4332]">
                  {selectedSign.name}
                </h2>
                <span className="bg-[#E8F5F1] text-[#1B5E4F] px-3 py-1 rounded-full text-xs font-medium">
                  {selectedSign.category}
                </span>
                <button
                  onClick={() => setSelectedSign(null)}
                  className="w-full mt-2 py-3 bg-gradient-to-r from-[#52B69A] to-[#1B5E4F] text-white rounded-full font-bold text-sm"
                >
                  إغلاق
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
