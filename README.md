# ₿ BTC_TFT_VMD_Dashboard_V2: Neural Trading Engine

**₿ BTC_TFT_VMD_Dashboard_V2: Neural Trading Engine** est une plateforme de trading algorithmique avancée combinant la **Décomposition en Modes Variationnels (VMD)** et les **Temporal Fusion Transformers (TFT)**. Le système est conçu pour prédire les mouvements du Bitcoin avec une précision institutionnelle en extrayant les tendances de fond du bruit de marché.


## 🧠 Spécifications du Modèle

* **Fenêtre d'entrée (Lookback)** : Le modèle analyse une séquence glissante de **168 heures** (7 jours) de données historiques pour comprendre le contexte temporel et les cycles hebdomadaires.
* **Fenêtre de sortie (Forecast)** : À partir de cet historique, il génère une séquence prédictive continue sur les **24 heures** suivantes.
* **Architecture Hybride** :
    * **VMD** : Nettoie le signal en isolant les composantes IMF (*Intrinsic Mode Functions*) pour réduire le bruit.
    * **TFT** : Utilise des mécanismes d'attention pour identifier les variables les plus influentes (RSI, Volatilité, Momentum) à chaque instant.

## 🚀 Options de Lancement

Le projet est entièrement conteneurisé avec **Docker**. Vous avez deux méthodes d'utilisation :

### Option A : Full Local (Mode Privé)
Idéal pour une utilisation autonome où le moteur d'IA tourne sur votre propre machine.
1. Assurez-vous que Docker est installé sur votre machine.
2. Ouvrez un terminal à la racine du projet.
3. Lancez la commande : `docker compose up --build`.
4. L'interface est disponible sur `http://localhost:8505`.

### Option B : Hybride (Frontend Local + Backend Cloud)
Utilise l'API haute performance hébergée sur **Google Cloud Platform**.
1. Dans le fichier `docker-compose.yml`, modifiez la variable `BACKEND_URL` pour pointer vers votre adresse Cloud Run :mentionnée dans le fichier même.
2. Relancez uniquement le frontend : `docker compose up --build frontend`.
*Cette option permet de déporter les calculs lourds (VMD/TFT) sur le Cloud tout en gardant une interface réactive en local.*

## 📈 Guide d'Utilisation du Dashboard

* **Import des données** : Chargez un fichier CSV contenant l'historique (format timestamp secondes) via le widget de téléchargement.
* **Configuration des repères** :
    * 🔴 **Passage de Mois** : Affiche un point rouge unique à chaque début de mois sur la courbe de prix.
    * 🟡 **Vendredi/Samedi** : Trace des lignes jaunes verticales pour marquer les clôtures hebdomadaires.
* **Visualisation** :
    * **Ligne Bleue** : Données d'observation réelles.
    * **Ligne Violette Pointillée** : Prédiction du modèle avec labels **H+1, H+2...**.
* **Interactivité** : Utilisez la molette pour le **Zoom** et le clic-glissé pour le **Pan** (déplacement).

## ⚖️ Matrice de Décision (Risk Management)

Le desk suit une logique de validation stricte basée sur la performance prédite et l'indice de confiance :

| Signal | Performance (Prix) | Confiance (Jauge) | Signification |
| :--- | :--- | :--- | :--- |
| **BUY** | > +0.15% | Haute (> 40%) | Opportunité d'achat validée |
| **SELL** | < -0.15% | Haute (> 40%) | Signal de baisse confirmé |
| **WAIT** | Abs > 0.15% | Basse (≤ 40%) | Tendance détectée mais trop risqué |
| **HOLD** | Entre ±0.15% | Indifférent | Marché latéral / sans direction |

## 🛠️ Stack Technique

* **Frontend** : Streamlit, Plotly.
* **Backend** : FastAPI, Uvicorn (avec support CORS pour le Cloud).
* **IA** : PyTorch Forecasting, Temporal Fusion Transformer.
* **Signal** : VMDpy (*Variational Mode Decomposition*).
# ₿ Alpha Forecast Desk : Neural Trading Engine

**Alpha Forecast Desk** est une plateforme de trading algorithmique avancée combinant la **Décomposition en Modes Variationnels (VMD)** et les **Temporal Fusion Transformers (TFT)**. Le système est conçu pour prédire les mouvements du Bitcoin avec une précision institutionnelle en extrayant les tendances de fond du bruit de marché.



## ⚠️ IMPORTANT : Structure du fichier CSV
Pour que le modèle puisse générer des prédictions, votre fichier CSV doit respecter strictement la structure suivante :

* **Colonne `timestamp` (float)** : Temps au format UNIX (secondes).
* **Colonne `high` (float)** : Prix le plus haut.
* **Colonne `close` (float)** : Prix de clôture.
* **Colonne `volume` (float)** : Volume de transaction.
* **Colonne `low` (float)** : Prix le plus bas.
* **Colonne `open` (float)** : Prix d'ouverture.

**Spécifications techniques :**
* **Intervalle** : Les données doivent avoir un pas de **1 minute** (60 secondes entre chaque timestamp).
* **Format** : Le fichier doit être un CSV avec séparateur **virgule**.

## 🧠 Spécifications du Modèle
* **Fenêtre d'entrée (Lookback)** : Le modèle analyse une séquence glissante de **168 heures** (7 jours) de données historiques pour comprendre le contexte actuel.
* **Fenêtre de sortie (Forecast)** : À partir de cet historique, il génère une séquence prédictive continue sur les **24 heures** suivantes.
* **Architecture Hybride** : 
    * **VMD** : Nettoie le signal en isolant les composantes IMF (*Intrinsic Mode Functions*).
    * **TFT** : Utilise des mécanismes d'attention pour identifier les variables les plus influentes.

## 📂 Arborescence du Projet (Package Tree)
Organisation des fichiers pour le déploiement :

```text
Alpha-Forecast-Desk/
├── .streamlit/
│   └── config.toml             # Configuration du thème sombre forcé
├── backend/
│   ├── main.py                 # API FastAPI avec middleware CORS
│   ├── Dockerfile              # Instructions d'image Backend
│   ├── requirements.txt        # torch, vmdpy, pytorch-forecasting...
│   └── V2-96N-isolated.ckpt    # Checkpoint du modèle TFT
├── frontend/
│   ├── app.py                  # Interface Streamlit & Plotly
│   ├── Dockerfile              # Instructions d'image Frontend
│   └── requirements.txt        # streamlit, pandas, requests, plotly...
├── docker-compose.yml          # Orchestration Docker
├── .gitignore                  # Fichiers exclus (venv, csv, logs)
└── README.md                   # Documentation du projet

---
**Développé par Mahmoud TOURKI** [LinkedIn](https://www.linkedin.com/in/mahmoud-tourki-b228b9147/) | [Email](mailto:mahmoud.tourki24@gmail.com)