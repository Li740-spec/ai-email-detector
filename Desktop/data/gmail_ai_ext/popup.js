document.getElementById('checkBtn').addEventListener('click', async () => {
  const btn = document.getElementById('checkBtn');
  const probText = document.getElementById('probText');
  const gaugeFill = document.getElementById('gaugeFill');
  const statusMsg = document.getElementById('statusMsg');
  const badgeContainer = document.getElementById('badgeContainer');

  // --- 重點：請將網址換成你 Render 的真實網址 ---
  const API_URL = 'https://ai-email-detector-1.onrender.com/predict';

  btn.disabled = true;
  btn.innerText = "掃描中...";
  statusMsg.innerText = "正在分析郵件語意...";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // 注入內容抓取
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => document.body.innerText
    });

    const emailContent = results[0].result;

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: emailContent })
    });

    if (!response.ok) throw new Error('API 回應錯誤');

    const data = await response.json();
    
    // UI 更新
    const prob = data.phish_prob || 0;
    probText.innerText = prob + '%';
    
    // 圓環動畫 (周長改為 251)
    const offset = 251 - (251 * prob / 100);
    gaugeFill.style.strokeDashoffset = offset;
    gaugeFill.style.stroke = prob > 50 ? "#ff4757" : "#2ed573";
    
    statusMsg.innerText = prob > 50 ? "🚩 高風險：偵測到釣魚威脅" : "✅ 郵件安全";
    statusMsg.style.color = prob > 50 ? "#ff4757" : "#2ed573";
    badgeContainer.innerHTML = (data.threat_category || []).map(t => `<span class="badge">${t}</span>`).join('');

  } catch (error) {
    console.error("Fetch Error:", error);
    statusMsg.innerText = "❌ 連線失敗，請檢查 API 網址";
    statusMsg.style.color = "red";
  } finally {
    btn.disabled = false;
    btn.innerText = "再次掃描";
  }
});