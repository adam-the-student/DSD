// dashboard.js
const logBox = document.getElementById('log-box');
const netStatus = document.getElementById('net-status');
const statusCard = document.getElementById('status-card');
const ledgerBody = document.getElementById('ledger-body');
const counterElement = document.getElementById('total-entries-count');

let ws;
let lastLoggedStatus = "";
let verifiedEntryCount = 0;

function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);

    ws.onopen = () => {
        netStatus.textContent = `Connected Live to Wall Pi Frame Server @ ${window.location.host}`;
        netStatus.style.color = "#50fa7b";
        addLog("System Link Established. Fetching data pipeline telemetry...");
    };

    ws.onclose = () => {
        netStatus.textContent = "Disconnected from Wall Pi Server. Retrying link...";
        netStatus.style.color = "#ff5555";
        addLog("SYSTEM WARNING: Network WebSocket Connection Interrupted. Reconnecting...");
        setTimeout(() => {
            connectWebSocket();
        }, 1000);
    };

    ws.onerror = (err) => {
        console.error("WebSocket transport error observed:", err);
        ws.close();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // 1. Process initial CSV database reload upon connection initialization
        if (data.is_startup_history && data.history) {
            ledgerBody.innerHTML = ""; 
            verifiedEntryCount = data.history.length;
            counterElement.textContent = verifiedEntryCount;
            
            data.history.forEach(item => {
                addLedgerRow(item.time, item.profile, item.confidence, item.proximity);
            });
            return; 
        }

        // 2. PRIORITIZE LIVE MILESTONE EVENTS: Process instantly and exit the function completely
        if (data.is_entry_event === true) {
            verifiedEntryCount++;
            counterElement.textContent = verifiedEntryCount;
            addLedgerRow(data.time, data.profile, data.confidence, data.proximity);
            addLog(`[ALERT] Real-time ledger updated: ${data.profile} (${data.confidence})`);
            return; // 🟢 Hard exit ensures we never touch the live HUD processing logic below!
        }

        // 3. Continuous real-time updates for the upper live HUD fields
        // 🟢 ULTRA SAFETY GUARD: Explicitly check that badge_status exists and is a string before running toUpperCase
        if (data && typeof data.badge_status === 'string') {
            document.getElementById('badge-status').textContent = data.badge_status.toUpperCase();
            
            if (typeof data.estimated_ft === 'number') {
                document.getElementById('distance-value').textContent = `${data.estimated_ft.toFixed(1)} ft`;
            }
            if (data.distance_status) {
                document.getElementById('zone-status').textContent = data.distance_status;
            }

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
        }
    };
}

function addLog(message) {
    const timestamp = new Date().toLocaleTimeString();
    logBox.innerHTML += `[${timestamp}] ${message}<br>`;
    logBox.scrollTop = logBox.scrollHeight;
}

function addLedgerRow(time, profile, confidence, distance) {
    const row = document.createElement('tr');
    
    // Strict compliance check
    const isCompliant = profile.startsWith("Valid") || profile.includes("Verified");
    
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

// Kickoff connection engine
connectWebSocket();