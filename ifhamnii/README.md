# 🤟 Ifhamnii - Plateforme d'Apprentissage de la Langue des Signes

Une application web moderne pour apprendre et explorer la langue des signes à travers une interface intuitive et interactive.

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Démarrage](#-démarrage)
- [Structure du projet](#-structure-du-projet)
- [Technologies](#-technologies)
- [Authentification](#-authentification)
- [Scripts disponibles](#-scripts-disponibles)
- [Contribution](#-contribution)

## ✨ Fonctionnalités

- 🔐 **Authentification sécurisée** avec Supabase
- 📹 **Mode caméra** pour pratiquer la reconnaissance des signes
- 📖 **Dictionnaire complet** avec vidéos des signes
- ⏱️ **Historique** des apprentissages
- 🎓 **Tutoriels interactifs** et tests
- ⚙️ **Paramètres utilisateur** personnalisables
- 🎨 **Interface moderne** avec Tailwind CSS
- 📱 **Responsive design** pour tous les appareils

### Catégories disponibles

- 🎨 Couleurs
- 🔢 Chiffres
- 🎓 Tests et certificats
- 📚 Éducation et apprentissage
- 👤 Traits personnels
- 😊 Émotions et sentiments
- 💼 Professions
- 🌍 Différentes langues
- 🎵 Musique

## 🔧 Prérequis

- Node.js 18+ ou pnpm 9+
- Un compte Supabase
- Git

## 📦 Installation

1. **Cloner le repository**

```bash
git clone <repository-url>
cd ifhamnii
```

2. **Installer les dépendances**

```bash
npm install
# ou avec pnpm
pnpm install
# ou avec yarn
yarn install
```

3. **Configuration des variables d'environnement**

```bash
cp .env.example .env.local
```

Remplir le fichier `.env.local` avec vos identifiants Supabase :

```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

## 🚀 Démarrage

### Développement

```bash
npm run dev
```

L'application s'ouvrira à `http://localhost:5173`

### Build production

```bash
npm run build
```

### Preview production

```bash
npm run preview
```

### Linting

```bash
npm run lint
```

## 📁 Structure du projet

```
src/
├── pages/                 # Pages principales
│   ├── Home.tsx          # Accueil
│   ├── Login.tsx         # Connexion
│   ├── Register.tsx      # Inscription
│   ├── Camera.tsx        # Mode caméra
│   ├── Dictionary.tsx    # Dictionnaire
│   ├── History.tsx       # Historique
│   ├── Tutorial.tsx      # Tutoriels
│   ├── Upload.tsx        # Upload de vidéos
│   ├── Settings.tsx      # Paramètres
│   └── Onboarding.tsx    # Onboarding
├── components/            # Composants réutilisables
│   ├── LogoMark.tsx
│   ├── ProtectedRoute.tsx
├── context/              # Context API
│   └── AuthContext.tsx
├── lib/                  # Utilitaires
│   ├── supabase.ts      # Client Supabase
│   └── theme.ts         # Configuration thème
├── assets/               # Assets statiques
├── App.tsx              # Composant principal
├── main.tsx             # Point d'entrée
└── index.css            # Styles globaux

public/signs/             # Vidéos des signes par catégorie
├── ألوان/
├── ارقام/
├── الإختبارات والشهادات/
└── ... (autres catégories)
```

## 🛠️ Technologies

- **Frontend Framework**: React 19.2.4
- **Language**: TypeScript 6.0.2
- **Build Tool**: Vite 8.0.4
- **Styling**: Tailwind CSS 4.2.2
- **Routing**: React Router 7.14.0
- **Backend/Auth**: Supabase 2.103.3
- **UI Components**: Lucide React 1.8.0
- **Animations**: Motion 12.38.0
- **Linting**: ESLint 9.39.4

## 🔐 Authentification

L'authentification est gérée via Supabase avec un contexte React (`AuthContext.tsx`).

- Inscription et connexion sécurisées
- Protection des routes avec `ProtectedRoute.tsx`
- Gestion de session automatique
- Support de multiple méthodes d'authentification

## 🎯 Scripts disponibles

| Commande          | Description                          |
| ----------------- | ------------------------------------ |
| `npm run dev`     | Lance le serveur développement       |
| `npm run build`   | Compile le projet pour la production |
| `npm run lint`    | Vérifie le code avec ESLint          |
| `npm run preview` | Prévisualise le build production     |

## 📝 Conventions de code

- **Composants**: Utiliser la syntaxe fonctionnelle avec hooks
- **Nommage**: camelCase pour les variables/fonctions, PascalCase pour les composants
- **Styles**: Utiliser les classes Tailwind CSS
- **Types**: Définir les types TypeScript pour toutes les props

## 🤝 Contribution

Les contributions sont bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Support

Pour toute question ou problème, veuillez ouvrir une issue dans le repository.

---

**Construit avec ❤️ pour l'accessibilité et l'inclusion**
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
globalIgnores(['dist']),
{
files: ['**/*.{ts,tsx}'],
extends: [
// Other configs...
// Enable lint rules for React
reactX.configs['recommended-typescript'],
// Enable lint rules for React DOM
reactDom.configs.recommended,
],
languageOptions: {
parserOptions: {
project: ['./tsconfig.node.json', './tsconfig.app.json'],
tsconfigRootDir: import.meta.dirname,
},
// other options...
},
},
])

```

```
