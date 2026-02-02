import os
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from vmdpy import VMD
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import uvicorn

# Force le mode CPU pour éviter les conflits de drivers sur le Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = FastAPI(title="Alpha Forecast Engine")

# --- AJOUT DU MIDDLEWARE CORS (Indispensable pour GCP) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Autorise les requêtes de votre Frontend local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TFT_CHECKPOINT_PATH = "V2-96N-isolated.ckpt"
ENCODER_LEN = 168

# Chargement du modèle au démarrage
model = None
if os.path.exists(TFT_CHECKPOINT_PATH):
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(TFT_CHECKPOINT_PATH, map_location=torch.device('cpu'))
        model.eval()
        print("✅ Modèle TFT chargé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")

def apply_vmd_causal(df_subset):
    if len(df_subset) < ENCODER_LEN: return df_subset
    signal = df_subset["close"].tail(ENCODER_LEN).values.astype(np.float32)
    last_p = signal[-1]
    try:
        # Décomposition VMD
        u, _, _ = VMD(signal, 2000, 0, 3, 0, 1, 1e-7)
        for i in range(3):
            val_imf = (u[i, -1] - last_p) / last_p
            df_subset.loc[df_subset.index[-1], f"IMF{i + 1}"] = np.clip(val_imf, -0.1, 0.1)
    except: 
        pass
    return df_subset

@app.post("/predict")
async def predict(data: dict):
    if model is None: 
        raise HTTPException(status_code=500, detail="Modèle introuvable sur le serveur")
    
    try:
        # Conversion du JSON en DataFrame
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # --- CALCULS TECHNIQUES (Identique au bloc initial) ---
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_return"].rolling(window=24).std()
        df["momentum"] = df["close"].pct_change(periods=24)
        
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Features temporelles (String pour correspondre aux embeddings du modèle)
        df["hour"] = df.index.hour.astype(str)
        df["day_of_week"] = df.index.dayofweek.astype(str)
        df["month"] = df.index.month.astype(str)
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(str)
        
        # Initialisation IMF pour VMD
        for i in range(1, 4): df[f"IMF{i}"] = 0.0
        df = apply_vmd_causal(df).dropna()
        
        if len(df) < ENCODER_LEN:
            raise ValueError(f"Séquence trop courte après calculs : {len(df)}h (Min {ENCODER_LEN}h requis)")

        # Préparation du DataSet pour le TFT
        df["group_id"], df["time_idx"] = "BTC", np.arange(len(df))
        ds = TimeSeriesDataSet.from_parameters(model.dataset_parameters, df, predict=True, stop_randomization=True)
        
        # Inférence
        with torch.no_grad():
            output = model.predict(ds.to_dataloader(train=False, batch_size=1), mode="raw")
            # Extraction des prédictions (log-returns à l'indice 3 selon votre structure)
            preds_ret = output["prediction"][0, :, 3].cpu().numpy().flatten()
        
        # Reconstruction des prix à partir des log-returns
        last_close = float(df.iloc[-1]["close"])
        median_prices = []
        curr = last_close
        for r in preds_ret:
            curr = curr * np.exp(np.clip(r, -0.02, 0.02))
            median_prices.append(float(curr))
            
        return {
            "median": median_prices, 
            "last_close": last_close, 
            "last_date": str(df.index[-1])
        }
        
    except Exception as e:
        print(f"🔥 Erreur interne : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "online", "model_loaded": model is not None}

if __name__ == "__main__":
    # Google Cloud utilise souvent le port 8080 par défaut
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)