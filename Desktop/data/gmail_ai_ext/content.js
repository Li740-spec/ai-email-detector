chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "get_gmail_content") {
        // Gmail 郵件內容的標準 Selector
        const emailBody = document.querySelector('.a3s.aiL');
        const text = emailBody ? emailBody.innerText : "";
        sendResponse({content: text});
    }
    return true;
});