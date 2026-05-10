import { useAuth } from "../context/AuthContext"
import { Navigate } from "react-router"

export default function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()

  // attend que la session soit chargée
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#E8F5F1] via-white to-[#E8F5F1]">
        <div className="w-12 h-12 border-4 border-[#52B69A] border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  // pas connecté → redirige vers login
  if (!user) return <Navigate to="/login" />

  return <>{children}</>
}