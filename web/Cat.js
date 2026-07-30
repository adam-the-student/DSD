// --- COLD CORE SOCKET INITIALIZATION FOR CAT VITAL MONITORS ---
let catSocket;

function connectCatWebSocket() {
    // Dynamically calculate paths using host constraints safely
    catSocket = new WebSocket(`ws://${window.location.host}/ws`);

    catSocket.onopen = () => {
        console.log("🔌 Connected to Tamagotchi Pet Telemetry stream link.");
        document.getElementById('vitals-mood-text').innerText = "System Link: Connected";
    };

    catSocket.onmessage = (event) => {
        const data = json.parse(event.data);
        
        // Skip operational boot arrays entirely on this specialized interface
        if (data.is_startup_history) return;

        // --- UPDATE DYNAMIC HUD READOUTS ---
        if (data.pet_status !== undefined) {
            document.getElementById('vitals-status').innerText = data.pet_status.toUpperCase();
        }
        if (data.successful_feedings !== undefined) {
            document.getElementById('vitals-feedings').innerText = data.successful_feedings;
        }
        if (data.daily_goal !== undefined) {
            document.getElementById('vitals-goal').innerText = data.daily_goal;
        }
        if (data.fps !== undefined) {
            document.getElementById('vitals-fps').innerText = `${data.fps} FPS`;
        }

        // --- ANIMATION GRAPHIC ROUTER ---
        updateCatAnimation(data.badge_status, data.pet_status);
    };

    catSocket.onclose = () => {
        console.log("❌ Lost network broadcast link. Attempting context reset pass...");
        document.getElementById('vitals-mood-text').innerText = "System Link: Disconnected";
        document.getElementById('vitals-mood-text').style.color = "#ff5555";
        setTimeout(connectCatWebSocket, 2000); // 2-second reconnect buffer
    };
}

// --- MASCOT SPRITE SELECTION ROUTER ---
function updateCatAnimation(systemStatus, petStatus) {
    const catImage = document.getElementById('retro-cat-mascot');
    const statusLabel = document.getElementById('vitals-mood-text');
    if (!catImage || !statusLabel) return;

    const normalizedStatus = (systemStatus || "").toUpperCase();
    const normalizedPet = (petStatus || "").toUpperCase();

    if (normalizedStatus.includes("INITIALIZING") || normalizedStatus.includes("SCANNING")) {
        catImage.src = "static/assets/booting.gif";
        statusLabel.innerText = "System Link: Calibrating...";
        statusLabel.style.color = "#8d8d99";
        return;
    }

    if (normalizedStatus.includes("NO BADGE") || normalizedStatus.includes("ALERT")) {
        catImage.src = "static/assets/shocked.gif";
        statusLabel.innerText = "System Link: Warning Issued";
        statusLabel.style.color = "#ff5555";
        return;
    }

    if (normalizedStatus.includes("BADGE DETECTED") || normalizedStatus.includes("DOS DETECTED")) {
        catImage.src = "static/assets/feeding.gif";
        statusLabel.innerText = "System Link: Feeding Confirmed!";
        statusLabel.style.color = "#00ff66";
        return;
    }

    if (normalizedPet === "DEAD" || normalizedPet === "SICK") {
        catImage.src = "static/assets/sick.gif";
        statusLabel.innerText = `System Link: Critical Health`;
        statusLabel.style.color = "#ff5555";
        return;
    }
    
    if (normalizedPet === "SATISFIED" || normalizedPet === "FULL") {
        catImage.src = "static/assets/happy.gif";
        statusLabel.innerText = "System Link: Target Satisfied";
        statusLabel.style.color = "#00ff66";
        return;
    }

    // Default monitoring state
    catImage.src = "static/assets/idle.gif";
    statusLabel.innerText = "System Link: Active";
    statusLabel.style.color = "#8d8d99";
}

// Boot socket loop instantly
connectCatWebSocket();