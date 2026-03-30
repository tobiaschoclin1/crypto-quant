# 🚀 Deployment Guide - Crypto Quant Pro

## Deploy to Render.com (FREE & Recommended)

Render.com offers **free hosting** for web services with the following benefits:
- ✅ Always-on (with occasional sleep after inactivity)
- ✅ Automatic HTTPS
- ✅ Auto-deploy from GitHub
- ✅ No credit card required
- ✅ 750 hours/month free

---

## 📋 Step-by-Step Deployment

### 1. **Create Render Account**

1. Go to [render.com](https://render.com)
2. Click **"Get Started for Free"**
3. Sign up with GitHub (recommended for easy deployment)

---

### 2. **Deploy from GitHub**

#### Option A: Using Blueprint (Easiest)

1. **Push your code** (already done ✓)
2. In Render dashboard, click **"New"** → **"Blueprint"**
3. Connect your GitHub repository: `tobiaschoclin1/crypto-quant`
4. Render will auto-detect `render.yaml`
5. Click **"Apply"**
6. Wait 2-3 minutes for deployment
7. Your app will be live at: `https://crypto-quant-pro.onrender.com`

#### Option B: Manual Setup

1. In Render dashboard, click **"New"** → **"Web Service"**
2. Connect your GitHub repository: `tobiaschoclin1/crypto-quant`
3. Configure:
   - **Name**: `crypto-quant-pro`
   - **Region**: Oregon (Free)
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend_api:app --host 0.0.0.0 --port $PORT`
4. Select **Free Plan**
5. Click **"Create Web Service"**
6. Wait for deployment

---

### 3. **Access Your Live App**

Once deployed, your app will be available at:
```
https://crypto-quant-pro.onrender.com
```

Or your custom URL if you configured one.

---

## 🔧 Alternative Free Hosting Options

### Option 2: Railway.app

**Pros**: $5 free credit/month, faster cold starts
**Cons**: Requires credit card after trial

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. New Project → Deploy from GitHub
4. Select `tobiaschoclin1/crypto-quant`
5. Railway auto-detects Python
6. Add environment variable: `PORT=8000`
7. Deploy

---

### Option 3: Fly.io

**Pros**: Global edge network, fast
**Cons**: More complex setup

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Deploy
flyctl launch
flyctl deploy
```

---

### Option 4: PythonAnywhere

**Pros**: Simple Python hosting
**Cons**: More manual configuration

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Clone your repo in console
3. Set up web app with WSGI
4. Configure to use `backend_api.py`

---

## ⚙️ Post-Deployment Configuration

### Keep the App Awake (Render Free Tier)

Render free tier sleeps after 15 min of inactivity. To keep it active:

**Option 1**: Use a cron service
- [cron-job.org](https://cron-job.org) - Free
- Set to ping your URL every 10 minutes

**Option 2**: Use UptimeRobot
1. Go to [uptimerobot.com](https://uptimerobot.com)
2. Add New Monitor
3. Monitor Type: HTTP(s)
4. URL: Your Render URL
5. Monitoring Interval: 5 minutes

---

## 🔒 Security Considerations

### For Production Use:

1. **Add Rate Limiting**:
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   ```

2. **Environment Variables**:
   - Add secrets in Render dashboard
   - Never commit API keys

3. **CORS Configuration**:
   - Update allowed origins in `backend_api.py`

---

## 📊 Monitoring

Once deployed, monitor your app:

- **Render Dashboard**: View logs, metrics, deployments
- **Health Check**: `https://your-app.onrender.com/analisis`
- **Logs**: Real-time in Render dashboard

---

## 🐛 Troubleshooting

### App won't start?
- Check logs in Render dashboard
- Verify `requirements.txt` has all dependencies
- Ensure Python version compatibility

### Slow response?
- Free tier may have cold starts (10-30s)
- First request after sleep takes longer
- Consider using UptimeRobot to keep awake

### Port errors?
- Render provides `$PORT` environment variable
- Make sure start command uses `--port $PORT`

---

## 💡 Tips for Free Hosting

1. **Auto-deploy**: Enable in Render to auto-deploy on git push
2. **Custom Domain**: Free on Render (add in settings)
3. **Logs**: Enable persistent logs for debugging
4. **Backup**: Keep your GitHub repo updated
5. **Monitoring**: Use free monitoring tools

---

## 📞 Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Community**: [Render Community](https://community.render.com)
- **GitHub Issues**: Report bugs in your repo

---

## ✅ Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Render account created
- [ ] Web service deployed
- [ ] App is accessible via URL
- [ ] Health check passes
- [ ] UptimeRobot configured (optional)
- [ ] Custom domain added (optional)

---

**Your app is now live 24/7 for FREE! 🎉**
