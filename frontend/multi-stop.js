// Multi-Stop Dashboard Functionality

let multiStopWs = null;
let currentMultiSession = null;
let multiStopData = {}; // stop_id -> {predictions: [], status: '', etc}

// Tab switching
function switchTab(tabName) {
    const tabs = document.querySelectorAll('.tab-button');
    const contents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(t => t.classList.remove('active'));
    contents.forEach(c => c.classList.remove('active'));
    
    event.target.classList.add('active');
    document.getElementById(`${tabName}-stop-tab`).classList.add('active');
    
    if (tabName === 'multi') {
        loadMultiStopTrips();
    }
}

// API functions for multi-stop
async function createMultiStopSession(tripId) {
    const response = await fetch(`${API_BASE}/multi-sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trip_id: tripId })
    });
    if (!response.ok) throw new Error('Failed to create multi-stop session');
    return await response.json();
}

async function getMultiStopSession(sessionId) {
    const response = await fetch(`${API_BASE}/multi-sessions/${sessionId}`);
    if (!response.ok) throw new Error('Failed to fetch multi-stop session');
    return await response.json();
}

async function getOverallMetrics(sessionId) {
    const response = await fetch(`${API_BASE}/multi-sessions/${sessionId}/metrics`);
    if (!response.ok) throw new Error('Failed to fetch overall metrics');
    return await response.json();
}

async function getStopMetrics(sessionId, stopId) {
    const response = await fetch(`${API_BASE}/multi-sessions/${sessionId}/stop/${stopId}/metrics`);
    if (!response.ok) throw new Error('Failed to fetch stop metrics');
    return await response.json();
}

async function stopMultiSession(sessionId) {
    const response = await fetch(`${API_BASE}/multi-sessions/${sessionId}`, {
        method: 'DELETE'
    });
    if (!response.ok) throw new Error('Failed to stop multi-session');
    return await response.json();
}

// UI functions
async function loadMultiStopTrips() {
    const tripsList = document.getElementById('multi-trips-list');
    tripsList.innerHTML = '<p class="loading">Loading trips...</p>';
    
    try {
        const trips = await fetchActiveTrips();
        
        if (trips.length === 0) {
            tripsList.innerHTML = '<p class="info-text">No active trips found.</p>';
            return;
        }
        
        tripsList.innerHTML = '';
        trips.forEach(trip => {
            const tripEl = document.createElement('div');
            tripEl.className = 'item';
            tripEl.innerHTML = `
                <div class="item-title">Trip ${trip.trip_id}</div>
                <div class="item-meta">
                    Route: ${trip.route_id || 'Unknown'} | Vehicle: ${trip.vehicle_id || 'Unknown'}
                </div>
            `;
            tripEl.addEventListener('click', () => startMultiStopSession(trip));
            tripsList.appendChild(tripEl);
        });
    } catch (error) {
        console.error('Failed to load trips:', error);
        tripsList.innerHTML = '<p class="error-text">Failed to load trips</p>';
    }
}

async function startMultiStopSession(trip) {
    try {
        const session = await createMultiStopSession(trip.trip_id);
        currentMultiSession = session;
        
        // Initialize stop data
        multiStopData = {};
        Object.keys(session.stops).forEach(stopId => {
            multiStopData[stopId] = {
                predictions: [],
                status: session.stops[stopId].status,
                stopInfo: session.stops[stopId]
            };
        });
        
        // Show session panel
        document.getElementById('multi-select-section').classList.add('hidden');
        document.getElementById('multi-session-panel').classList.remove('hidden');
        
        // Set session header
        document.getElementById('multi-session-title').textContent = 
            `Trip ${trip.trip_id} - ${Object.keys(session.stops).length} Stops`;
        document.getElementById('multi-session-meta').textContent = 
            `Route ${trip.route_id} | Vehicle ${trip.vehicle_id}`;
        
        // Render stop cards
        renderStopsGrid(session.stops);
        
        // Connect WebSocket
        connectMultiStopWebSocket(session.session_id);
        
    } catch (error) {
        console.error('Failed to start multi-stop session:', error);
        alert('Failed to start tracking session');
    }
}

function renderStopsGrid(stops) {
    const grid = document.getElementById('multi-stops-grid');
    grid.innerHTML = '';
    
    // Sort by stop_sequence
    const sortedStops = Object.entries(stops).sort((a, b) => 
        a[1].stop_sequence - b[1].stop_sequence
    );
    
    sortedStops.forEach(([stopId, stop]) => {
        const card = createStopCard(stopId, stop);
        grid.appendChild(card);
    });
}

function createStopCard(stopId, stopInfo) {
    const card = document.createElement('div');
    card.className = 'stop-card';
    card.id = `stop-card-${stopId}`;
    card.dataset.stopId = stopId;
    card.dataset.status = stopInfo.status;
    
    card.innerHTML = `
        <div class="stop-card-header">
            <h4>#${stopInfo.stop_sequence} ${stopInfo.stop_name}</h4>
            <span class="stop-status status-${stopInfo.status}">${stopInfo.status}</span>
        </div>
        <div class="stop-card-body">
            <div class="stop-eta" id="stop-eta-${stopId}">
                <div class="eta-label">Current ETA</div>
                <div class="eta-value">--:--</div>
            </div>
            <div class="stop-stats" id="stop-stats-${stopId}">
                <div class="stat">
                    <span class="stat-label">Predictions</span>
                    <span class="stat-value" id="stop-pred-count-${stopId}">0</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Distance</span>
                    <span class="stat-value" id="stop-distance-${stopId}">--</span>
                </div>
            </div>
            <div class="stop-error hidden" id="stop-error-${stopId}">
                <div class="error-label">Prediction Error</div>
                <div class="error-value"></div>
            </div>
        </div>
    `;
    
    return card;
}

function connectMultiStopWebSocket(sessionId) {
    const wsUrl = `ws://${window.location.host}/ws/multi/${sessionId}`;
    multiStopWs = new WebSocket(wsUrl);
    
    multiStopWs.onopen = () => {
        console.log('Multi-stop WebSocket connected');
    };
    
    multiStopWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMultiStopUpdate(data);
    };
    
    multiStopWs.onerror = (error) => {
        console.error('Multi-stop WebSocket error:', error);
    };
    
    multiStopWs.onclose = () => {
        console.log('Multi-stop WebSocket closed');
        multiStopWs = null;
    };
}

function handleMultiStopUpdate(data) {
    const { type, stop_id } = data;
    
    if (type === 'prediction') {
        handlePredictionUpdate(stop_id, data.data);
    } else if (type === 'arrival') {
        handleArrivalUpdate(stop_id, data.data);
    } else if (type === 'session_status') {
        handleSessionStatusUpdate(data.data);
    }
}

function handlePredictionUpdate(stopId, predData) {
    // Store prediction
    if (!multiStopData[stopId]) return;
    multiStopData[stopId].predictions.push(predData);
    
    // Update UI
    const card = document.getElementById(`stop-card-${stopId}`);
    if (!card) return;
    
    // Update ETA
    const etaValue = card.querySelector('.eta-value');
    const etaTime = new Date(predData.predicted_arrival);
    etaValue.textContent = etaTime.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
    
    // Update stats
    document.getElementById(`stop-pred-count-${stopId}`).textContent = 
        multiStopData[stopId].predictions.length;
    document.getElementById(`stop-distance-${stopId}`).textContent = 
        `${Math.round(predData.distance_meters)}m`;
    
    // Highlight card with animation
    card.classList.add('updated');
    setTimeout(() => card.classList.remove('updated'), 1000);
}

async function handleArrivalUpdate(stopId, arrivalData) {
    console.log('Arrival detected:', stopId, arrivalData);
    
    // Update status
    if (multiStopData[stopId]) {
        multiStopData[stopId].status = 'arrived';
    }
    
    const card = document.getElementById(`stop-card-${stopId}`);
    if (!card) return;
    
    // Update status badge
    const statusBadge = card.querySelector('.stop-status');
    statusBadge.textContent = 'arrived';
    statusBadge.className = 'stop-status status-arrived';
    card.dataset.status = 'arrived';
    
    // Fetch and display metrics
    try {
        const metrics = await getStopMetrics(currentMultiSession.session_id, stopId);
        displayStopMetrics(stopId, metrics);
    } catch (error) {
        console.error('Failed to fetch stop metrics:', error);
    }
    
    // Check if all stops are complete
    checkOverallCompletion();
}

function displayStopMetrics(stopId, metrics) {
    const card = document.getElementById(`stop-card-${stopId}`);
    if (!card) return;
    
    const errorDiv = document.getElementById(`stop-error-${stopId}`);
    errorDiv.classList.remove('hidden');
    
    const errorValue = errorDiv.querySelector('.error-value');
    const finalError = metrics.final_error_seconds;
    const mae = metrics.mae_seconds;
    
    errorValue.innerHTML = `
        <div class="error-stat">
            <span>Final Error:</span>
            <span class="${finalError > 0 ? 'error-late' : 'error-early'}">
                ${formatError(finalError)}
            </span>
        </div>
        <div class="error-stat">
            <span>MAE:</span>
            <span>${(mae / 60).toFixed(1)} min</span>
        </div>
    `;
}

async function checkOverallCompletion() {
    const allArrived = Object.values(multiStopData).every(s => s.status === 'arrived');
    
    if (allArrived && currentMultiSession) {
        // Fetch overall metrics
        try {
            const metrics = await getOverallMetrics(currentMultiSession.session_id);
            displayOverallMetrics(metrics);
        } catch (error) {
            console.error('Failed to fetch overall metrics:', error);
        }
    }
}

function displayOverallMetrics(metrics) {
    const panel = document.getElementById('multi-overall-metrics');
    const content = document.getElementById('multi-metrics-content');
    
    panel.classList.remove('hidden');
    
    content.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Predictions</div>
                <div class="metric-value">${metrics.n_predictions_total}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Overall MAE</div>
                <div class="metric-value">${(metrics.overall_mae_minutes || 0).toFixed(1)} min</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mean Bias</div>
                <div class="metric-value">
                    ${(metrics.overall_mean_error_minutes || 0).toFixed(1)} min
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Stops Tracked</div>
                <div class="metric-value">${metrics.n_stops_completed} / ${metrics.n_stops_tracked}</div>
            </div>
        </div>
        <div class="per-stop-summary">
            <h4>Per-Stop Summary</h4>
            <table>
                <thead>
                    <tr>
                        <th>Stop</th>
                        <th>Predictions</th>
                        <th>MAE (min)</th>
                        <th>Final Error (min)</th>
                    </tr>
                </thead>
                <tbody>
                    ${metrics.stops.map(s => `
                        <tr>
                            <td>${s.stop_name}</td>
                            <td>${s.n_predictions}</td>
                            <td>${(s.mae_seconds / 60).toFixed(1)}</td>
                            <td class="${s.final_error_seconds > 0 ? 'error-late' : 'error-early'}">
                                ${(s.final_error_seconds / 60).toFixed(1)}
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function handleSessionStatusUpdate(statusData) {
    console.log('Session status:', statusData);
    
    if (statusData.status === 'completed') {
        // Session complete
        if (multiStopWs) {
            multiStopWs.close();
        }
    }
}

async function handleStopMultiSession() {
    if (!currentMultiSession) return;
    
    try {
        await stopMultiSession(currentMultiSession.session_id);
        
        if (multiStopWs) {
            multiStopWs.close();
            multiStopWs = null;
        }
        
        resetMultiStopUI();
    } catch (error) {
        console.error('Failed to stop multi-session:', error);
        alert('Failed to stop session');
    }
}

function resetMultiStopUI() {
    currentMultiSession = null;
    multiStopData = {};
    
    document.getElementById('multi-select-section').classList.remove('hidden');
    document.getElementById('multi-session-panel').classList.add('hidden');
    document.getElementById('multi-overall-metrics').classList.add('hidden');
    
    loadMultiStopTrips();
}

// Event listeners
document.getElementById('stop-multi-session-btn')?.addEventListener('click', handleStopMultiSession);
