# Run Instructions

## Quick Start

1. Create/activate the virtual environment (PowerShell):
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. **Run the Automation Pipeline**:
   To fetch the latest data, train the model, and launch the app:
   ```bash
   python update_pipeline.py
   ```

### Pipeline Options

| Command | Description |
|---------|-------------|
| `python update_pipeline.py` | Full pipeline: fetch → train → serve |
| `python update_pipeline.py --no-fetch` | Skip data fetch, run training and serve |
| `python update_pipeline.py --quick` | Quick launch: run inference and serve only |
| `python update_pipeline.py --quick --no-serve` | Update history log without launching server |
| `python -m src.app` | Launch web server directly |

*Note: The full pipeline fetches data from the FPL API and Understat. The first run may take a few minutes.*

**Configuration**: The pipeline defaults to the **2025** season for Understat data. Override this by editing `update_pipeline.py` or `src/data_fetch.py` if needed.

---

## Local Development

### Running the App
```bash
# Full pipeline (recommended for first run)
python update_pipeline.py

# Quick launch for development
python update_pipeline.py --quick
```

The app will be available at: http://127.0.0.1:5000

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Main predictions page |
| `/feature-importance` | Feature importance dashboard |
| `/api/predictions` | JSON predictions data |
| `/api/history` | Historical predictions with actual points |
| `/api/live?gw=N` | Live match scores for gameweek N |
| `/api/stats` | Usage statistics |

---

## Deployment

### Option 1: Northflank (Recommended)

The app is currently deployed on [Northflank](https://northflank.com/) with a custom domain.

1. **Create Northflank Service**:
   - Sign up and create a new **Combined Service**
   - Connect your GitHub repository
   - Select the branch to deploy

2. **Configure Build**:
   - **Build Type**: Buildpack
   - **Run Command**: `gunicorn src.app:app`
   - **Port**: `8080` (or as configured)

3. **Environment Variables** (if needed):
   - No additional environment variables required for basic deployment

4. **Custom Domain Setup** (with Cloudflare):
   - In Northflank, go to **Networking** → **Public** → **Add Domain**
   - Add your domain (e.g., `fplbangers.com`)
   - In Cloudflare, create a CNAME record pointing to your Northflank endpoint
   - Set Cloudflare SSL/TLS mode to **Full** or **Full (Strict)**
   - Disable Cloudflare proxy (orange cloud → grey) for initial SSL provisioning

5. **Deploy**: Push to your connected branch to trigger automatic deployment

### Option 2: Render (Free Tier)

[Render](https://render.com) offers a free tier with automatic SSL.

1. **Push to GitHub**: Ensure your code (including `data/processed` and `models`) is pushed.

2. **Create Web Service**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click **New +** → **Web Service**
   - Connect your GitHub repository

3. **Configure**:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn src.app:app`

4. **Deploy**: Click **Create Web Service**

---

## Updating the Live Site

To update the site with new predictions:

1. Run the pipeline locally:
   ```bash
   python update_pipeline.py
   ```

2. Commit and push the updated `data/` and `models/` folders:
   ```bash
   git add data/ models/
   git commit -m "Update predictions for GW XX"
   git push
   ```

3. The deployment platform will automatically redeploy with new data.

---

## Testing

### Unit Tests
```bash
python -m pytest tests/ -v
```

### Smoke Test
`src/test_api.py` contains a lightweight smoke test that hits each data source and prints samples.

```bash
python -m src.test_api
```

### Performance Test
```bash
python test_app_performance.py
```

---

## Troubleshooting

### Common Issues

1. **"Data not found" error**:
   - Run the full pipeline: `python update_pipeline.py`
   - Ensure `data/processed/inference_data.csv` exists

2. **Model loading errors**:
   - Retrain models: `python update_pipeline.py --no-fetch`
   - Check `models/` directory for `.pkl` files

3. **Port already in use**:
   - Change port: modify `app.run(port=5001)` in `src/app.py`
   - Or kill the existing process

4. **Northflank deployment fails**:
   - Check build logs in Northflank dashboard
   - Verify `Procfile` exists with: `web: gunicorn src.app:app`
   - Ensure port matches Northflank configuration
