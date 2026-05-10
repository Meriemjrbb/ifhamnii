import { useState } from "react"
import { useNavigate } from "react-router"
import { AnimatePresence, motion } from "motion/react"
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Camera,
  CheckCircle2,
  History,
  Lightbulb,
  Upload,
} from "lucide-react"
import LogoMark from "../components/LogoMark"

const STEPS = [
  {
    icon: Camera,
    label: "الكاميرا",
    title: "ترجمة مباشرة",
    description:
      "افتحي الكاميرا واجعلي اليدين واضحتين داخل الإطار. سيعرض التطبيق النص الناتج مباشرة حتى تراجعيه قبل الحفظ.",
    tip: "الإضاءة الجيدة والخلفية الهادئة تساعدان على نتيجة أوضح.",
    color: "from-[#1B5E4F] to-[#52B69A]",
  },
  {
    icon: Upload,
    label: "الفيديو",
    title: "رفع فيديو",
    description:
      "اختاري فيديو جاهزا من جهازك، ثم اتركي التطبيق يحلله ويحوّل الإشارات إلى نص قابل للنسخ أو الحفظ.",
    tip: "اختاري فيديو قصيرا وواضحا عندما تريدين نتيجة أسرع.",
    color: "from-[#40916C] to-[#1B4332]",
  },
  {
    icon: BookOpen,
    label: "القاموس",
    title: "قاموس الإشارات",
    description:
      "استخدمي القاموس لتصفّح الإشارات حسب الفئات، أو ابحثي عن كلمة معينة عندما تحتاجين مراجعة سريعة.",
    tip: "البحث هو أسرع طريق عندما تعرفين الكلمة التي تريدينها.",
    color: "from-[#74C69D] to-[#2D6A4F]",
  },
  {
    icon: History,
    label: "السجل",
    title: "سجل الترجمات",
    description:
      "كل ترجمة تحفظينها ستظهر في السجل، مع وقتها وخيارات النسخ والحذف حتى تعودي إليها بسهولة.",
    tip: "احفظي الترجمات المهمة بعد مراجعتها حتى تبقى مرتبة في السجل.",
    color: "from-[#1B4332] to-[#081C15]",
  },
]

export default function Tutorial() {
  const navigate = useNavigate()
  const [step, setStep] = useState(0)

  const current = STEPS[step]
  const Icon = current.icon
  const isLast = step === STEPS.length - 1
  const isFirst = step === 0

  return (
    <div dir="rtl" className="min-h-screen bg-[#F6FBF8]">
      <div className="mx-auto w-full max-w-5xl">
        <div className="bg-gradient-to-br from-[#D8F3DC] via-white to-[#B7E4C7] px-5 pb-8 pt-8 shadow-sm sm:px-8 lg:rounded-b-[32px]">
          <div className="mb-7 flex items-center justify-between">
            <button
              onClick={() => navigate("/settings")}
              className="flex h-10 w-10 items-center justify-center rounded-full bg-white text-[#1B5E4F] shadow-md"
            >
              <ArrowRight size={18} />
            </button>

            <button
              onClick={() => navigate("/home")}
              className="rounded-full bg-white/80 px-4 py-2 text-sm font-bold text-[#52796F] shadow-sm"
            >
              تخطي
            </button>
          </div>

          <div className="flex items-center gap-4">
            <LogoMark size="lg" />
            <div>
              <p className="text-sm font-bold text-[#52796F]">دليل سريع</p>
              <h1 className="text-3xl font-black text-[#1B4332]">كيف تستخدمين إفهمني</h1>
            </div>
          </div>

          <div className="mt-7 flex gap-2">
            {STEPS.map((item, index) => (
              <button
                key={item.label}
                onClick={() => setStep(index)}
                className={`h-2 flex-1 rounded-full transition-all ${index <= step ? "bg-[#1B5E4F]" : "bg-white/80"
                  }`}
                aria-label={item.label}
              />
            ))}
          </div>
        </div>

        <main className="px-5 py-6 sm:px-8">
          <div className="mb-5 flex gap-2 overflow-x-auto pb-1">
            {STEPS.map((item, index) => (
              <button
                key={item.label}
                onClick={() => setStep(index)}
                className={`shrink-0 rounded-full px-4 py-2 text-sm font-bold transition-colors ${index === step
                    ? "bg-[#1B5E4F] text-white shadow-md"
                    : "bg-white text-[#52796F] ring-1 ring-[#D8F3DC]"
                  }`}
              >
                {item.label}
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            <motion.section
              key={current.label}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.25 }}
              className="overflow-hidden rounded-[32px] bg-white shadow-sm ring-1 ring-[#E8F5F1]"
            >
              <div className={`bg-gradient-to-br ${current.color} px-6 py-8 text-white`}>
                <div className="mb-8 flex items-center justify-between">
                  <div className="rounded-full bg-white/15 px-4 py-2 text-sm font-bold">
                    {current.label}
                  </div>
                  <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white/15">
                    <Icon size={30} />
                  </div>
                </div>

                <h2 className="text-3xl font-black">{current.title}</h2>
                <p className="mt-3 text-sm leading-7 text-white/90">{current.description}</p>
              </div>

              <div className="p-5">
                <div className="flex items-start gap-3 rounded-2xl bg-[#E8F5F1] p-4">
                  <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-white text-[#1B5E4F]">
                    <Lightbulb size={18} />
                  </div>
                  <p className="text-sm font-bold leading-6 text-[#1B4332]">{current.tip}</p>
                </div>

                <div className="mt-5 flex items-center gap-3 rounded-2xl bg-[#F6FBF8] p-4 text-[#52796F]">
                  <CheckCircle2 className="shrink-0 text-[#52B69A]" size={20} />
                  <p className="text-sm font-bold">يمكنك الرجوع لهذا الدليل من صفحة الإعدادات.</p>
                </div>
              </div>
            </motion.section>
          </AnimatePresence>

          <div className="mt-6 flex gap-3">
            {!isFirst && (
              <motion.button
                whileTap={{ scale: 0.97 }}
                onClick={() => setStep(value => value - 1)}
                className="flex items-center gap-2 rounded-full bg-white px-5 py-4 font-bold text-[#1B5E4F] shadow-sm ring-1 ring-[#D8F3DC]"
              >
                <ArrowLeft size={18} />
                السابق
              </motion.button>
            )}

            <motion.button
              whileTap={{ scale: 0.97 }}
              onClick={() => (isLast ? navigate("/home") : setStep(value => value + 1))}
              className="flex flex-1 items-center justify-center gap-2 rounded-full bg-gradient-to-r from-[#52B69A] to-[#1B5E4F] py-4 font-bold text-white shadow-lg"
            >
              {isLast ? "ابدأ الاستخدام" : "التالي"}
              {!isLast && <ArrowRight size={18} />}
            </motion.button>
          </div>
        </main>
      </div>
    </div>
  )
}
