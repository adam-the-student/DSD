// --- COLD CORE SOCKET INITIALIZATION FOR CAT VITAL MONITORS ---
let catSocket;

// --- RETRO IDLE MIXER CONFIGURATION ---
const IDLE_BASE = "GIFS/nukoCatSniff.gif"; // Main sniffing loop
const IDLE_FLAIRS = [
    "GIFS/nukoCatPounce.gif" // Periodic pounce action
];

let currentSystemState = "IDLE";
let flairTimeoutToken = null;

// 🟢 HELPER FUNCTION: Safely sets image source and dynamically toggles the flip class
function changeMascotSource(newSrc) {
    const catImage = document.getElementById('retro-cat-mascot');
    if (!catImage) return;

    catImage.src = newSrc;

    // Only apply horizontal flip style if the asset is explicitly the sniffing animation
    if (newSrc === IDLE_BASE) {
        catImage.classList.add('flipped-cat');
    } else {
        catImage.classList.remove('flipped-cat');
    }
}

function connectCatWebSocket() {
    catSocket = new WebSocket(`ws://${window.location.host}/ws`);

    catSocket.onopen = () => {
        console.log("🔌 Connected to Tamagotchi Pet Telemetry stream link.");
        const statusLabel = document.getElementById('vitals-mood-text');
        if (statusLabel) {
            statusLabel.innerText = "SYSTEM LINK: CONNECTED";
        }
    };

    catSocket.onmessage = (event) => {
        let data;
        try {
            data = JSON.parse(event.data);
        } catch (e) {
            console.log("Parsing skipped: running in static mock view mode.");
            return;
        }
        
        if (!data || data.is_startup_history) return;

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

        updateCatAnimation(data.badge_status, data.pet_status);
    };

    catSocket.onclose = () => {
        console.log("❌ Lost network broadcast link. Attempting context reset pass...");
        const statusLabel = document.getElementById('vitals-mood-text');
        if (statusLabel) {
            statusLabel.innerText = "SYSTEM LINK: DISCONNECTED";
            statusLabel.style.color = "#ff5555";
        }
        setTimeout(connectCatWebSocket, 2000);
    };
}

function startIdleHeartbeatLoop() {
    clearTimeout(flairTimeoutToken);
    const randomInterval = Math.floor(Math.random() * (10000 - 4000) + 4000);

    flairTimeoutToken = setTimeout(() => {
        if (currentSystemState === "IDLE") {
            const chosenFlair = IDLE_FLAIRS[Math.floor(Math.random() * IDLE_FLAIRS.length)];
            changeMascotSource(chosenFlair);

            setTimeout(() => {
                if (currentSystemState === "IDLE") {
                    changeMascotSource(IDLE_BASE);
                    startIdleHeartbeatLoop();
                }
            }, 2500);
        } else {
            startIdleHeartbeatLoop();
        }
    }, randomInterval);
}

// --- MASCOT SPRITE SELECTION ROUTER ---
function updateCatAnimation(systemStatus, petStatus) {
    const statusLabel = document.getElementById('vitals-mood-text');
    if (!statusLabel) return;

    const normalizedStatus = (systemStatus || "").toUpperCase();
    const normalizedPet = (petStatus || "").toUpperCase();

    // --- INTERCEPT EVENT CONSTRAINTS ---
    if (normalizedStatus.includes("INITIALIZING") || normalizedStatus.includes("SCANNING")) {
        currentSystemState = "BUSY";
        changeMascotSource("GIFS/nukoWorriedQuestion.gif");
        statusLabel.innerText = "SYSTEM LINK: CALIBRATING...";
        return;
    }

    if (normalizedStatus.includes("NO BADGE") || normalizedStatus.includes("ALERT")) {
        currentSystemState = "BUSY";
        changeMascotSource("GIFS/nukoEmbarrassedDrop.gif");
        statusLabel.innerText = "SYSTEM LINK: WARNING ISSUED";
        return;
    }

    if (normalizedStatus.includes("BADGE DETECTED") || normalizedStatus.includes("DOS DETECTED")) {
        currentSystemState = "BUSY";
        changeMascotSource("GIFS/nukoCatHeart.gif");
        statusLabel.innerText = "SYSTEM LINK: FEEDING CONFIRMED!";
        
        setTimeout(() => {
            currentSystemState = "IDLE";
            changeMascotSource(IDLE_BASE);
            startIdleHeartbeatLoop();
        }, 4000);
        return;
    }

    // --- FALLBACK BASELINE IDLE CONDITIONS ---
    if (currentSystemState !== "BUSY") {
        currentSystemState = "IDLE";
        statusLabel.innerText = "SYSTEM LINK: ACTIVE";
        
        if (normalizedPet === "DEAD" || normalizedPet === "SICK") {
            changeMascotSource("GIFS/nukoEmbarrassedDrop.gif");
            statusLabel.innerText = "SYSTEM LINK: CRITICAL HEALTH";
        } else if (normalizedPet === "SATISFIED" || normalizedPet === "FULL") {
            changeMascotSource("GIFS/nukoCatHeart.gif");
            statusLabel.innerText = "SYSTEM LINK: TARGET SATISFIED";
        }
    }
}

connectCatWebSocket();
startIdleHeartbeatLoop();