# CryptoPredictions

A **kitchen-sink lab** for algorithmic crypto-asset forecasting.  
Each sub-folder is a self-contained technique— from classic tree ensembles to deep-learning feeds—ready to clone, tweak and deploy.

---

## 📂 Current modules

| Folder | Technique | Brief |
|--------|-----------|-------|
| `TriBoost/` | LightGBM + CatBoost + XGBoost stack | 8-hour SOL/USDC price forecasts with a FastAPI micro-service & Docker setup. |
| *(more coming)* | — | Reserve a folder and follow the pattern below. |

---

## 🔧 How to add your own model

1. **Create a folder** (`MyCoolModel/`).  
2. Include at minimum:  
   * `utils.py` – feature engineering  
   * `train.py` / `service.py` – training or inference endpoint  
   * `requirements.txt` – module-specific deps  
3. Add a short `README.md` inside the folder.  
4. Open a PR—describe what’s novel and any dataset needs.

> **Tip:** mirror `TriBoost/`’s structure for an easy starting point.

---

## 🛠️ Quick start (TriBoost demo)

```bash
cd TriBoost
docker compose up --build   # spins up FastAPI on :4562
# or run locally
pip install -r requirements.txt
python FitModelPredictManual.py
````

Swagger UI at **`/docs`** gives you live predictions.

---

## 🤝 Contributing

Bug fixes, new models, and dataset connectors are welcome!
Open an issue or Pull Request—let’s push crypto-forecasting research forward together.

---

⚡ Fueling public goods with [\$IACS](https://dexscreener.com/base/0xd4d742cc8f54083f914a37e6b0c7b68c6005a024) — Get involved: 0x46e69Fa9059C3D5F8933CA5E993158568DC80EBf (Base)
