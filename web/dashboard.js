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
            return;
        }

        // --- STEP 3: DYNAMIC BASELINE HUD READOUT EXTENSION ---
        if (data && data.daily_goal !== undefined) {
            // 🟢 FIXED: This now runs ALWAYS so your green text tracker never goes blank!
            const liveReadout = document.getElementById('current-json-baseline');
            if (liveReadout) {
                liveReadout.innerText = data.daily_goal;
            }
            
            // Only update the actual input typing box if it's completely empty (like on a fresh page load)
            const managerInput = document.getElementById('managerGoalInput');
            if (managerInput && !managerInput.value) {
                managerInput.value = data.daily_goal;
            }
        }

        // 4. Continuous real-time updates for the upper live HUD fields
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

// --- CONTEXT MANAGER SENDER FUNCTION LINKED TO CLICK EVENTS ---
function updateDailyGoal(rawInputValue) {
    // Parse the incoming argument value directly 
    const newGoal = parseInt(rawInputValue, 10);
    
    // Boundary check validation pass
    if (isNaN(newGoal) || newGoal < 1) {
        addLog("[SYS] Configuration Error: Please enter a valid goal number.");
        return;
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ "set_daily_goal": newGoal }));
        addLog(`[SYS] Sent target goal shift request: ${newGoal}`);
    } else {
        addLog("[SYS] Error updating configuration: WebSocket transport stream link is offline.");
    }
}
// Kickoff connection engine
connectWebSocket();