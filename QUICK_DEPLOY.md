# ⚡ Quick Deploy Guide - Get Live in 5 Minutes!

## 🎯 Fastest Way to Deploy (Render.com)

### Step 1: Create Account
👉 Go to **[render.com](https://render.com)** and click **"Get Started"**
- Sign up with GitHub (easiest option)

### Step 2: Deploy with Blueprint
1. Click **"New"** → **"Blueprint"**
2. Select repository: **`tobiaschoclin1/crypto-quant`**
3. Click **"Apply"**
4. ✅ Done! Wait 2-3 minutes

### Step 3: Access Your Live App
Your app will be at: **`https://crypto-quant-pro.onrender.com`**

---

## 🎨 What You Get

✅ **24/7 availability** - Always online (sleeps after 15min inactivity on free tier)
✅ **Auto HTTPS** - Secure by default
✅ **Auto-deploy** - Pushes to GitHub auto-deploy
✅ **Free forever** - No credit card needed
✅ **Custom domain** - Add your own domain (optional)

---

## ⚡ Keep It Awake (Optional)

Free tier sleeps after inactivity. To keep it active 24/7:

**Option 1: UptimeRobot (Recommended)**
1. Go to [uptimerobot.com](https://uptimerobot.com)
2. Create free account
3. Add New Monitor:
   - Type: HTTP(s)
   - URL: `https://crypto-quant-pro.onrender.com/analisis?symbol=BTCUSDT`
   - Interval: 5 minutes
4. ✅ Your app stays awake!

**Option 2: Cron-job.org**
1. Go to [cron-job.org](https://cron-job.org)
2. Create free account
3. Add cron job to ping your URL every 10 minutes

---

## 📱 Share Your App

Once deployed, share these URLs:

**Main App**: `https://crypto-quant-pro.onrender.com`
**API Endpoint**: `https://crypto-quant-pro.onrender.com/analisis?symbol=BTCUSDT`
**Backtest**: `https://crypto-quant-pro.onrender.com/backtest?symbol=ETHUSDT`

---

## 🔧 Troubleshooting

**Q: App not loading?**
- Wait 30 seconds on first request (cold start)
- Check Render dashboard for deployment status

**Q: Want faster load times?**
- Use UptimeRobot to prevent sleep
- Or upgrade to paid plan ($7/month)

**Q: Need help?**
- Check full guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- Render docs: [render.com/docs](https://render.com/docs)

---

## 🚀 That's It!

Your crypto trading app is now **LIVE and FREE**! 🎉

Next steps:
- ✅ Test the live URL
- ✅ Set up UptimeRobot (optional)
- ✅ Share with friends
- ✅ Enjoy 24/7 trading signals!
