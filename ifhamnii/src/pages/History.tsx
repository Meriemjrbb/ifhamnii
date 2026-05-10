import { useEffect, useMemo, useState } from "react"
import { useNavigate } from "react-router"
import { motion } from "motion/react"
import {
  ArrowRight,
  CalendarDays,
  Clock,
  Copy,
  Loader,
  Search,
  Trash2,
  X,
} from "lucide-react"
import LogoMark from "../components/LogoMark"
import { supabase } from "../lib/supabase"
import { useAuth } from "../context/AuthContext"

type HistoryItem = {
  id: string
  text: string
  duration: string
  created_at: string
}

export default function History() {
  const navigate = useNavigate()
  const { user } = useAuth()
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [copied, setCopied] = useState<string | null>(null)
  const [query, setQuery] = useState("")

  useEffect(() => {
    if (!user) return

    supabase
      .from("history")
      .select("*")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .then(({ data }) => {
        setHistory(data || [])
        setLoading(false)
      })
  }, [user])

  const copyText = (id: string, text: string) => {
    navigator.clipboard.writeText(text)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }

  const deleteItem = async (id: string) => {
    await supabase.from("history").delete().eq("id", id)
    setHistory(prev => prev.filter(item => item.id !== id))
  }

  const clearAll = async () => {
    if (!user) return

    await supabase.from("history").delete().eq("user_id", user.id)
    setHistory([])
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString("ar-TN-u-nu-latn", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    })
  }

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleTimeString("ar-TN-u-nu-latn", {
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  const filteredHistory = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()
    if (!normalizedQuery) return history

    return history.filter(item => item.text.toLowerCase().includes(normalizedQuery))
  }, [history, query])

  const grouped = useMemo(() => {
    return filteredHistory.reduce((acc, item) => {
      const date = item.created_at.split("T")[0]
      if (!acc[date]) acc[date] = []
      acc[date].push(item)
      return acc
    }, {} as Record<string, HistoryItem[]>)
  }, [filteredHistory])

  const latestItem = history[0]

  return (
    <div dir="rtl" className="min-h-screen bg-[#F6FBF8]">
      <div className="mx-auto w-full max-w-5xl">
        <div className="bg-gradient-to-br from-[#D8F3DC] via-white to-[#B7E4C7] px-5 pb-7 pt-8 shadow-sm sm:px-8 lg:rounded-b-[32px]">
          <div className="mb-6 flex items-center justify-between">
            <button
              onClick={() => navigate("/home")}
              className="flex h-10 w-10 items-center justify-center rounded-full bg-white text-[#1B5E4F] shadow-md"
            >
              <ArrowRight size={18} />
            </button>

            {history.length > 0 && (
              <button
                onClick={clearAll}
                className="rounded-full bg-white/80 px-4 py-2 text-sm font-bold text-red-400 shadow-sm"
              >
                مسح الكل
              </button>
            )}
          </div>

          <div className="flex items-center gap-4">
            <LogoMark size="lg" />
            <div>
              <p className="text-sm font-bold text-[#52796F]">ترجماتك المحفوظة</p>
              <h1 className="text-3xl font-black text-[#1B4332]">السجل</h1>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-3">
            <div className="rounded-2xl bg-white/80 p-4 shadow-sm">
              <p className="text-xs font-bold text-[#52796F]">عدد الترجمات</p>
              <p className="mt-1 text-2xl font-black text-[#1B5E4F]">{history.length}</p>
            </div>
            <div className="rounded-2xl bg-white/80 p-4 shadow-sm">
              <p className="text-xs font-bold text-[#52796F]">آخر ترجمة</p>
              <p className="mt-1 truncate text-sm font-black text-[#1B4332]">
                {latestItem ? formatTime(latestItem.created_at) : "لا يوجد"}
              </p>
            </div>
          </div>
        </div>

        <main className="px-5 py-5 sm:px-8">
          {history.length > 0 && (
            <div className="mb-5 flex items-center gap-3 rounded-2xl bg-white px-4 py-3 shadow-sm ring-1 ring-[#D8F3DC]">
              <Search className="text-[#52B69A]" size={18} />
              <input
                value={query}
                onChange={event => setQuery(event.target.value)}
                placeholder="ابحث في السجل"
                className="min-w-0 flex-1 bg-transparent text-right text-[#1B4332] outline-none placeholder:text-[#52796F]/60"
              />
              {query && (
                <button
                  onClick={() => setQuery("")}
                  className="flex h-7 w-7 items-center justify-center rounded-full bg-[#E8F5F1] text-[#52796F]"
                >
                  <X size={14} />
                </button>
              )}
            </div>
          )}

          {loading && (
            <div className="flex justify-center py-24">
              <Loader className="animate-spin text-[#52B69A]" size={40} />
            </div>
          )}

          {!loading && history.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-col items-center justify-center py-24 text-center"
            >
              <LogoMark size="xl" className="mb-5" />
              <p className="text-2xl font-black text-[#1B4332]">السجل فارغ</p>
              <p className="mt-2 max-w-xs text-sm leading-6 text-[#52796F]">
                ستظهر هنا الترجمات التي تحفظينها من الكاميرا أو رفع الفيديو.
              </p>
              <motion.button
                whileTap={{ scale: 0.97 }}
                onClick={() => navigate("/camera")}
                className="mt-7 rounded-full bg-gradient-to-r from-[#52B69A] to-[#1B5E4F] px-8 py-3 font-bold text-white shadow-lg"
              >
                ابدأ الترجمة
              </motion.button>
            </motion.div>
          )}

          {!loading && history.length > 0 && filteredHistory.length === 0 && (
            <div className="rounded-3xl bg-white p-8 text-center shadow-sm">
              <p className="font-bold text-[#1B4332]">لا توجد نتائج</p>
              <p className="mt-2 text-sm text-[#52796F]">جرّبي كلمة أخرى في البحث.</p>
            </div>
          )}

          {!loading &&
            Object.entries(grouped)
              .sort(([a], [b]) => b.localeCompare(a))
              .map(([date, items]) => (
                <section key={date} className="mb-7">
                  <div className="mb-3 flex items-center gap-2 text-[#52796F]">
                    <CalendarDays size={16} />
                    <p className="text-sm font-black">{formatDate(date)}</p>
                  </div>

                  <div className="flex flex-col gap-3">
                    {items.map((item, index) => (
                      <motion.article
                        key={item.id}
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.04 }}
                        className="rounded-3xl bg-white p-4 shadow-sm ring-1 ring-[#E8F5F1]"
                      >
                        <p className="text-lg font-black leading-8 text-[#1B4332]">{item.text}</p>

                        <div className="mt-4 flex items-center justify-between gap-3 border-t border-[#E8F5F1] pt-3">
                          <div className="flex min-w-0 items-center gap-2 text-[#52796F]">
                            <Clock size={15} />
                            <p className="truncate text-xs font-bold">
                              {formatTime(item.created_at)}
                              {item.duration && ` · ${item.duration}`}
                            </p>
                          </div>

                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => copyText(item.id, item.text)}
                              className="flex h-9 w-9 items-center justify-center rounded-full bg-[#E8F5F1] text-[#1B5E4F]"
                            >
                              {copied === item.id ? (
                                <span className="text-xs font-black">تم</span>
                              ) : (
                                <Copy size={15} />
                              )}
                            </button>

                            <button
                              onClick={() => deleteItem(item.id)}
                              className="flex h-9 w-9 items-center justify-center rounded-full bg-red-50 text-red-400"
                            >
                              <Trash2 size={15} />
                            </button>
                          </div>
                        </div>
                      </motion.article>
                    ))}
                  </div>
                </section>
              ))}
        </main>
      </div>
    </div>
  )
}
