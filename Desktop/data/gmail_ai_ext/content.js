chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "get_gmail_content") {
        // 嘗試多種 Gmail 內文選擇器
        const selectors = ['.a3s.aiL', '.ii.gt', '.adn.ads', '[role="main"]'];
        let content = "";
        
        for (let s of selectors) {
            let el = document.querySelector(s);
            if (el && el.innerText.length > 5) {
                content = el.innerText;
                break;
            }
        }
        sendResponse({ content: content });
    }
    return true;
});