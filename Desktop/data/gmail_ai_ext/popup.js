const API_BASE = 'https://ai-email-detector-iuns.onrender.com';
let lastCapturedText = "";

document.addEventListener('DOMContentLoaded', function() {
    const btn = document.getElementById('detect-btn');
    const res = document.getElementById('result');
    const farea = document.getElementById('feedback-area');

    btn.addEventListener('click', () => {
        res.innerHTML = "🔍 正在讀取內容並喚醒 AI...";
        farea.style.display = "none";

        chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
            if (!tabs[0]) return;
            chrome.tabs.sendMessage(tabs[0].id, {action: "get_gmail_content"}, (response) => {
                
                if (!response || !response.content) {
                    res.innerHTML = "<b style='color:orange'>⚠️ 找不到內容！<br>請先點開一封郵件</b>";
                    return;
                }

                lastCapturedText = response.content;
                res.innerHTML = "🚀 分析中...<br><small>首次偵測約需 30 秒</small>";

                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 60000);

                fetch(`${API_BASE}/predict`, {
                    method: 'POST',
                    mode: 'cors',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: lastCapturedText}),
                    signal: controller.signal
                })
                .then(r => r.json())
                .then(data => {
                    clearTimeout(timeoutId);
                    let color = data.label === 'phishing' ? '#c0392b' : '#27ae60';
                    let statusText = data.label === 'phishing' ? '🚩 警告：這是釣魚郵件！' : '✅ 安全：這是一般郵件';
                    
                    res.innerHTML = `
                        <b style="color: ${color}; font-size: 15px;">${statusText}</b>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: ${data.phish_prob}%; background: ${color};"></div>
                        </div>
                        <div style="font-size: 12px; display: flex; justify-content: space-between;">
                            <span>風險值: ${data.phish_prob}%</span>
                            <span>關鍵特徵: ${data.keywords.length > 0 ? data.keywords.join(', ') : '無'}</span>
                        </div>
                    `;
                    farea.style.display = "block";
                })
                .catch(e => {
                    clearTimeout(timeoutId);
                    res.innerHTML = "<span style='color:red'>❌ 連線超時<br>請確認伺服器已啟動後重試</span>";
                });
            });
        });
    });

    document.getElementById('fb-safe').onclick = () => sendFb('safe');
    document.getElementById('fb-phish').onclick = () => sendFb('phishing');

    function sendFb(l) {
        fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            mode: 'cors',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: lastCapturedText, label: l})
        }).then(() => alert("感謝回饋！AI 模型將在下次更新中學習。"));
    }
});