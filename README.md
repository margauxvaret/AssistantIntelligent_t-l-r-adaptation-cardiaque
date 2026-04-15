# 🫀 Cardio-Réadap Pro

Application de **téléréadaptation cardiaque assistée par IA**, conçue pour tourner sur Google Colab avec un GPU gratuit.

## Fonctionnalités

- **Tableau de bord patient** — bilan cardiaque, simulation d'effort, métriques en temps réel
- **Calendrier** — suivi des séances de réadaptation et consultations
- **Discussion IA** — chatbot médical basé sur Mistral-7B-Instruct (quantifié 4-bit)
- Interface sombre professionnelle (design médical épuré)

## 🚀 Démarrage rapide sur Google Colab

### 1. Ouvrir le notebook de lancement

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VOTRE_USERNAME/cardio-readap-pro/blob/main/colab_launch.ipynb)

> **⚠️ Important :** Activez un GPU avant de démarrer  
> *Environnement d'exécution → Modifier le type → GPU (T4)*

### 2. Obtenir un token ngrok

Créez un compte gratuit sur [ngrok.com](https://ngrok.com) et copiez votre token depuis le [dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).

### 3. Exécuter les cellules dans l'ordre

| Cellule | Action |
|---------|--------|
| 1 | Installation des dépendances |
| 2 | Clonage du repo |
| 3 | Lancement de l'app + tunnel ngrok |
| 4 *(optionnel)* | Upload de documents médicaux |

Un lien public (ex: `https://xxxx.ngrok.io`) s'affiche — cliquez dessus pour accéder à l'app.

## 📁 Structure du projet

```
cardio-readap-pro/
├── app.py                  # Application Streamlit principale
├── colab_launch.ipynb      # Notebook de démarrage Colab
├── requirements.txt        # Dépendances Python
├── documents_cardio/       # Vos documents médicaux (ignoré par git)
└── README.md
```

## ⚙️ Stack technique

| Composant | Technologie |
|-----------|------------|
| Interface | Streamlit |
| IA / LLM | Mistral-7B-Instruct (4-bit via bitsandbytes) |
| Embeddings | sentence-transformers |
| Recherche vectorielle | FAISS |
| Graphiques | Plotly |
| Tunnel | ngrok |

## 🔒 Sécurité

- **Ne commitez jamais** votre token ngrok dans le code
- Les documents médicaux sont dans `.gitignore`
- Les poids du modèle sont téléchargés à la volée depuis HuggingFace (non stockés dans le repo)

## 📋 Patients de démonstration

L'application inclut 3 profils patients fictifs :
- Patient 1 — Post-infarctus (phase de consolidation)
- Patient 2 — Insuffisance cardiaque (phase initiale)
- Patient 3 — Post-opératoire chirurgie cardiaque (phase avancée)
