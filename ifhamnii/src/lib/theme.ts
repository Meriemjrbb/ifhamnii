export const DARK_MODE_KEY = "ifhamnii-dark-mode"

export function readSavedDarkMode() {
  try {
    return localStorage.getItem(DARK_MODE_KEY) === "true"
  } catch {
    return false
  }
}

export function applyDarkMode(enabled: boolean) {
  document.documentElement.dataset.theme = enabled ? "dark" : "light"
}

export function saveDarkMode(enabled: boolean) {
  try {
    localStorage.setItem(DARK_MODE_KEY, String(enabled))
  } catch {
    // Ignore storage errors; the current page can still update visually.
  }

  applyDarkMode(enabled)
}
