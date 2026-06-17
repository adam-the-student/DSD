// dashboard.js
const logBox = document.getElementById('log-box');
const netStatus = document.getElementById('net-status');
const statusCard = document.getElementById('status-card');
const ledgerBody = document.getElementById('ledger-body');
const counterElement = document.getElementById('total-entries-count');

const ws = new WebSocket(`ws://${window.location.host}/ws`);
let lastLoggedStatus = "";
let verifiedEntryCount = 0;

ws.onopen = () => {
    netStatus.textContent = `Connected Live to Wall Pi Frame Server @ ${window.location.host}`;
    netStatus.style.color = "#50fa7b";
    addLog("System Link Established. Fetching data pipeline telemetry...");
};

ws.onclose = () => {
    netStatus.textContent = "Disconnected from Wall Pi Server. Retrying link...";
    netStatus.style.color = "#ff5555";
    addLog("SYSTEM WARNING: Network WebSocket Connection Interrupted.");
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.is_startup_history && data.history) {
        ledgerBody.innerHTML = ""; 
        verifiedEntryCount = data.history.length;
        counterElement.textContent = verifiedEntryCount;
        
        data.history.forEach(item => {
            addLedgerRow(item.time, item.profile, item.confidence, item.proximity);
        });
        return; 
    }

    document.getElementById('badge-status').textContent = data.badge_status.toUpperCase();
    document.getElementById('distance-value').textContent = `${data.estimated_ft.toFixed(1)} ft`;
    document.getElementById('zone-status').textContent = data.distance_status;

    statusCard.className = "card full-width"; 
    if (data.distance_status !== "OK" && data.distance_status !== "OFFLINE") {
        statusCard.classList.add("state-ALERT");
    } else if (data.badge_status.includes("BADGE DETECTED")) {
        statusCard.classList.add("state-BADGE");
    } else if (data.badge_status.includes("NO BADGE DETECTED")) {
        statusCard.classList.add("state-NO-BADGE");
    } else {
        statusCard.classList.add("state-SCANNING");
    }

    if (data.badge_status !== lastLoggedStatus && !data.badge_status.includes("SCANNING")) {
        addLog(`[DIAG] State shift: ${data.badge_status}`);
        lastLoggedStatus = data.badge_status;
    }

    if (data.is_entry_event === true) {
        verifiedEntryCount++;
        counterElement.textContent = verifiedEntryCount;
        addLedgerRow(data.event_time, data.event_profile, data.max_confidence, data.min_distance);
    }
};

function addLog(message) {
    const timestamp = new Date().toLocaleTimeString();
    logBox.innerHTML += `[${timestamp}] ${message}<br>`;
    logBox.scrollTop = logBox.scrollHeight;
}

function addLedgerRow(time, profile, confidence, distance) {
    const row = document.createElement('tr');
    
    const isCompliant = profile.includes("YES") || profile.includes("Valid");
    const badgeClass = isCompliant ? "badge-tag state-BADGE" : "badge-tag state-NO-BADGE";
    const labelText = isCompliant ? "✅ Dosimeter Verified" : "❌ No Dosimeter";

    row.innerHTML = `
        <td>${time}</td>
        <td><span class="${badgeClass}" style="padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: bold;">${labelText}</span></td>
        <td>${confidence}</td>
        <td>${distance}</td>
    `;
    ledgerBody.insertBefore(row, ledgerBody.firstChild);
}