import { createBrowserRouter, RouterProvider } from "react-router"
import Onboarding from "./pages/Onboarding"
import Login from "./pages/Login"
import Register from "./pages/Register"
import Home from "./pages/Home"
import Camera from "./pages/Camera"
import Upload from "./pages/Upload"
import Dictionary from "./pages/Dictionary"
import History from "./pages/History"
import Settings from "./pages/Settings"
import Tutorial from "./pages/Tutorial"
import ProtectedRoute from "./components/ProtectedRoute"

const router = createBrowserRouter([
  { path: "/",           element: <Onboarding /> },
  { path: "/login",      element: <Login /> },
  { path: "/register",   element: <Register /> },
  { path: "/home",       element: <ProtectedRoute><Home /></ProtectedRoute> },
  { path: "/camera",     element: <ProtectedRoute><Camera /></ProtectedRoute> },
  { path: "/upload",     element: <ProtectedRoute><Upload /></ProtectedRoute> },
  { path: "/dictionary", element: <ProtectedRoute><Dictionary /></ProtectedRoute> },
  { path: "/history",    element: <ProtectedRoute><History /></ProtectedRoute> },
  { path: "/settings",   element: <ProtectedRoute><Settings /></ProtectedRoute> },
  { path: "/tutorial",   element: <ProtectedRoute><Tutorial /></ProtectedRoute> },
])

export default function App() {
  return <RouterProvider router={router} />
}
// App.tsx est le composant racine — c'est lui qui orchestre tout. Mettre le routeur ici c'est la convention la plus propre. Quand on ajoutera des écrans, on les ajoutera tous ici.