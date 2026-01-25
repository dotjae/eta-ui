// Live ETA Prediction & Validation UI - Frontend Application

const API_BASE = window.location.origin + '/api';
let ws = null;
let currentSession = null;
let selectedRoute = null;
let selectedTrip = null;
let selectedStop = null;

// Utility functions
function formatTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

function formatDuration(seconds) {
    const mins = Math.floor(Math.abs(seconds) / 60);
    const secs = Math.floor(Math.abs(seconds) % 60);
    const sign = seconds < 0 ? '-' : '';
    return `${sign}${mins}m ${secs}s`;
}

function formatError(seconds) {
    const mins = (seconds / 60).toFixed(1);
    if (seconds > 0) {
        return `+${mins} min (predicted late)`;
    } else {
        return `${mins} min (predicted early)`;
    }
}

// API functions
async function fetchActiveRoutes() {
    const response = await fetch(`${API_BASE}/routes`);
    if (!response.ok) throw new Error('Failed to fetch routes');
    return await response.json();
}

async function fetchActiveTrips(routeId = null) {
    const url = routeId ? `${API_BASE}/trips?route_id=${encodeURIComponent(routeId)}` : `${API_BASE}/trips`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch trips');
    return await response.json();
}

async function fetchTripStops(tripId) {
    const response = await fetch(`${API_BASE}/trips/${tripId}/stops?future_only=true`);
    if (!response.ok) throw new Error('Failed to fetch stops');
    return await response.json();
}

async function createSession(tripId, stopId) {
    const response = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trip_id: tripId, stop_id: stopId })
    });
    if (!response.ok) throw new Error('Failed to create session');
    return await response.json();
}

async function getSessionEvaluation(sessionId) {
    const response = await fetch(`${API_BASE}/sessions/${sessionId}/evaluation`);
    if (!response.ok) throw new Error('Failed to fetch evaluation');
    return await response.json();
}

async function stopSession(sessionId) {
    const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
        method: 'DELETE'
    });
    if (!response.ok) throw new Error('Failed to stop session');
    return await response.json();
}

// UI rendering functions
function renderRoutes(routes) {
    const routesList = document.getElementById('routes-list');
    routesList.innerHTML = '';

    if (routes.length === 0) {
        routesList.innerHTML = '<p class="info-text">No active routes found. Please try again in a moment.</p>';
        return;
    }

    routes.forEach(route => {
        const routeEl = document.createElement('div');
        routeEl.className = 'item';
        routeEl.innerHTML = `
            <div class="item-title">Route ${route.route_id}</div>
            <div class="item-meta">
                <span>${route.active_trips} active trip${route.active_trips !== 1 ? 's' : ''}</span>
            </div>
        `;

        routeEl.addEventListener('click', () => selectRoute(route));
        routesList.appendChild(routeEl);
    });
}

function renderTrips(trips) {
    const tripsList = document.getElementById('trips-list');
    tripsList.innerHTML = '';

    if (trips.length === 0) {
        tripsList.innerHTML = '<p class="info-text">No active trips found for this route.</p>';
        return;
    }

    trips.forEach(trip => {
        const tripEl = document.createElement('div');
        tripEl.className = 'item';
        tripEl.innerHTML = `
            <div class="item-title">Trip ${trip.trip_id}</div>
            <div class="item-subtitle">Vehicle ${trip.vehicle_id || 'Unknown'}</div>
            <div class="item-meta">
                <span>Last seen: ${trip.last_seen ? formatTime(trip.last_seen) : 'Unknown'}</span>
                ${trip.current_stop_sequence ? `<span>Stop ${trip.current_stop_sequence}</span>` : ''}
            </div>
        `;

        tripEl.addEventListener('click', () => selectTrip(trip));
        tripsList.appendChild(tripEl);
    });
}

function renderStops(stops) {
    const stopsList = document.getElementById('stops-list');
    stopsList.innerHTML = '';

    if (stops.length === 0) {
        stopsList.innerHTML = '<p class="info-text">No future stops for this trip.</p>';
        return;
    }

    stops.forEach(stop => {
        const stopEl = document.createElement('div');
        stopEl.className = 'item';
        stopEl.innerHTML = `
            <div class="item-title">${stop.stop_name}</div>
            <div class="item-subtitle">Stop ID: ${stop.stop_id}</div>
            <div class="item-meta">
                <span>Sequence: ${stop.stop_sequence}</span>
                ${stop.arrival_time ? `<span>Scheduled: ${stop.arrival_time}</span>` : ''}
            </div>
        `;

        stopEl.addEventListener('click', () => selectStop(stop));
        stopsList.appendChild(stopEl);
    });
}

function updatePredictionDisplay(prediction) {
    const etaValue = document.getElementById('current-eta-value');
    const etaDistance = document.getElementById('current-eta-distance');

    etaValue.textContent = formatDuration(prediction.eta_seconds);
    etaDistance.textContent = `${Math.round(prediction.distance_meters)} meters away`;
}

function addPredictionToTimeline(prediction) {
    const timeline = document.getElementById('predictions-list');

    const item = document.createElement('div');
    item.className = 'timeline-item';

    // Build feature details HTML
    const features = prediction.features || {};
    const modelScope = prediction.model_scope || 'unknown';
    const modelKey = prediction.model_key || 'unknown';

    const detailsId = `details-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    item.innerHTML = `
        <div class="timeline-item-time">${formatTime(prediction.timestamp)}</div>
        <div class="timeline-item-eta">ETA: ${formatDuration(prediction.eta_seconds)}</div>
        <div class="timeline-item-meta">
            ${Math.round(prediction.distance_meters)}m away • Model: ${prediction.model_type || 'unknown'}
            <button class="details-toggle" onclick="toggleDetails('${detailsId}')">
                <span class="toggle-icon">▶</span> Details
            </button>
        </div>
        <div id="${detailsId}" class="prediction-details" style="display: none;">
            <div class="details-section">
                <div class="details-header">Model Information</div>
                <div class="details-grid">
                    <div class="detail-item">
                        <span class="detail-label">Key:</span>
                        <span class="detail-value">${modelKey}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Type:</span>
                        <span class="detail-value">${prediction.model_type || 'unknown'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Scope:</span>
                        <span class="detail-value badge badge-${modelScope}">${modelScope}</span>
                    </div>
                </div>
            </div>
            <div class="details-section">
                <div class="details-header">Temporal Features</div>
                <div class="details-grid">
                    <div class="detail-item">
                        <span class="detail-label">Hour:</span>
                        <span class="detail-value">${features.hour ?? 'N/A'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Day of Week:</span>
                        <span class="detail-value">${features.day_of_week ?? 'N/A'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Peak Hour:</span>
                        <span class="detail-value">${features.is_peak_hour ? 'Yes' : 'No'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Weekend:</span>
                        <span class="detail-value">${features.is_weekend ? 'Yes' : 'No'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Holiday:</span>
                        <span class="detail-value">${features.is_holiday ? 'Yes' : 'No'}</span>
                    </div>
                </div>
            </div>
            <div class="details-section">
                <div class="details-header">Spatial Features</div>
                <div class="details-grid">
                    <div class="detail-item">
                        <span class="detail-label">Distance:</span>
                        <span class="detail-value">${Math.round(features.distance_to_stop || 0)}m</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Stop Seq:</span>
                        <span class="detail-value">${features.stop_sequence ?? 'N/A'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Segment Progress:</span>
                        <span class="detail-value">${((features.progress_on_segment || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Route Progress:</span>
                        <span class="detail-value">${((features.progress_ratio || 0) * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
            <div class="details-section">
                <div class="details-header">Weather Features</div>
                <div class="details-grid">
                    <div class="detail-item">
                        <span class="detail-label">Temperature:</span>
                        <span class="detail-value">${features.temperature_c ?? 'N/A'}°C</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Precipitation:</span>
                        <span class="detail-value">${features.precipitation_mm ?? 'N/A'}mm</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Wind Speed:</span>
                        <span class="detail-value">${features.wind_speed_kmh ?? 'N/A'} km/h</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    timeline.insertBefore(item, timeline.firstChild);
}

function toggleDetails(detailsId) {
    const details = document.getElementById(detailsId);
    const toggleIcon = event.target.closest('.details-toggle').querySelector('.toggle-icon');

    if (details.style.display === 'none') {
        details.style.display = 'block';
        toggleIcon.textContent = '▼';
    } else {
        details.style.display = 'none';
        toggleIcon.textContent = '▶';
    }
}

function showEvaluation(metrics) {
    document.getElementById('prediction-section').style.display = 'none';
    document.getElementById('evaluation-section').style.display = 'block';

    document.getElementById('actual-arrival-time').textContent = formatTime(metrics.actual_arrival);
    document.getElementById('total-predictions').textContent = metrics.n_predictions;

    const finalError = document.getElementById('final-error');
    finalError.textContent = formatError(metrics.final_error_seconds);
    finalError.className = 'metric-value ' + (metrics.final_error_seconds > 0 ? 'error-positive' : 'error-negative');

    const mae = document.getElementById('mae');
    mae.textContent = `${metrics.mae_minutes.toFixed(2)} min`;

    const meanError = document.getElementById('mean-error');
    meanError.textContent = formatError(metrics.mean_error_seconds);
    meanError.className = 'metric-value ' + (metrics.mean_error_seconds > 0 ? 'error-positive' : 'error-negative');

    // Draw chart
    drawErrorChart(metrics.predictions);
}

function drawErrorChart(predictions) {
    const canvas = document.getElementById('error-chart-canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = 300;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (predictions.length === 0) {
        ctx.fillStyle = '#6b7280';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No predictions to display', width / 2, height / 2);
        return;
    }

    // Extract data
    const times = predictions.map(p => p.time_to_arrival_seconds);
    const errors = predictions.map(p => p.error_seconds);

    const maxTime = Math.max(...times);
    const minTime = Math.min(...times);
    const maxError = Math.max(...errors.map(Math.abs));

    // Draw axes
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;

    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();

    // Zero line
    ctx.strokeStyle = '#9ca3af';
    ctx.setLineDash([5, 5]);
    const zeroY = height - padding - (maxError > 0 ? (height - 2 * padding) / 2 : 0);
    ctx.beginPath();
    ctx.moveTo(padding, zeroY);
    ctx.lineTo(width - padding, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Plot points
    ctx.fillStyle = '#2563eb';
    predictions.forEach(p => {
        const x = padding + ((maxTime - p.time_to_arrival_seconds) / (maxTime - minTime)) * (width - 2 * padding);
        const y = zeroY - (p.error_seconds / (maxError || 1)) * ((height - 2 * padding) / 2);

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
    });

    // Draw line connecting points
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2;
    ctx.beginPath();
    predictions.forEach((p, i) => {
        const x = padding + ((maxTime - p.time_to_arrival_seconds) / (maxTime - minTime)) * (width - 2 * padding);
        const y = zeroY - (p.error_seconds / (maxError || 1)) * ((height - 2 * padding) / 2);

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';

    // X-axis label
    ctx.fillText('Time to Arrival (seconds)', width / 2, height - 10);

    // Y-axis label
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Prediction Error (seconds)', 0, 0);
    ctx.restore();
}

// Event handlers
async function loadRoutes() {
    const btn = document.getElementById('load-routes-btn');
    const status = document.getElementById('routes-status');

    btn.disabled = true;
    status.innerHTML = '<span class="loading"></span> Loading routes...';

    try {
        const routes = await fetchActiveRoutes();
        renderRoutes(routes);
        status.textContent = `Found ${routes.length} active route${routes.length !== 1 ? 's' : ''}`;
    } catch (error) {
        console.error('Failed to load routes:', error);
        status.textContent = 'Error loading routes. Please try again.';
    } finally {
        btn.disabled = false;
    }
}

async function selectRoute(route) {
    selectedRoute = route;

    // Highlight selected route
    document.querySelectorAll('#routes-list .item').forEach(el => {
        el.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');

    // Load trips for this route
    try {
        const trips = await fetchActiveTrips(route.route_id);
        renderTrips(trips);
        document.getElementById('trip-section').style.display = 'block';
    } catch (error) {
        console.error('Failed to load trips:', error);
        alert('Failed to load trips for this route');
    }
}

async function selectTrip(trip) {
    selectedTrip = trip;

    // Highlight selected trip
    document.querySelectorAll('#trips-list .item').forEach(el => {
        el.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');

    // Load stops for this trip
    try {
        const stops = await fetchTripStops(trip.trip_id);
        renderStops(stops);
        document.getElementById('stop-section').style.display = 'block';
    } catch (error) {
        console.error('Failed to load stops:', error);
        alert('Failed to load stops for this trip');
    }
}

async function selectStop(stop) {
    selectedStop = stop;

    // Highlight selected stop
    document.querySelectorAll('#stops-list .item').forEach(el => {
        el.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');

    // Start prediction session
    try {
        const session = await createSession(selectedTrip.trip_id, stop.stop_id);
        currentSession = session;

        // Update UI
        document.getElementById('session-trip-id').textContent = session.trip_id;
        document.getElementById('session-stop-name').textContent = session.stop_name;
        document.getElementById('session-status').textContent = 'Active';
        document.getElementById('session-status').className = 'badge badge-active';

        // Show prediction section
        document.getElementById('prediction-section').style.display = 'block';
        document.getElementById('predictions-list').innerHTML = '';

        // Connect WebSocket
        connectWebSocket(session.session_id);
    } catch (error) {
        console.error('Failed to start session:', error);
        alert('Failed to start prediction session');
    }
}

function connectWebSocket(sessionId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'prediction') {
            updatePredictionDisplay(message.data);
            addPredictionToTimeline(message.data);
        } else if (message.type === 'arrival') {
            handleArrival();
        } else if (message.type === 'status') {
            console.log('Session status:', message.data);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
    };
}

async function handleArrival() {
    if (ws) {
        ws.close();
        ws = null;
    }

    document.getElementById('session-status').textContent = 'Arrived';
    document.getElementById('session-status').className = 'badge badge-arrived';

    // Fetch evaluation results
    try {
        const metrics = await getSessionEvaluation(currentSession.session_id);
        showEvaluation(metrics);
    } catch (error) {
        console.error('Failed to load evaluation:', error);
        alert('Failed to load evaluation results');
    }
}

async function handleStopSession() {
    if (!currentSession) return;

    try {
        await stopSession(currentSession.session_id);

        if (ws) {
            ws.close();
            ws = null;
        }

        // Reset UI
        resetUI();
    } catch (error) {
        console.error('Failed to stop session:', error);
        alert('Failed to stop session');
    }
}

function resetUI() {
    currentSession = null;
    selectedRoute = null;
    selectedTrip = null;
    selectedStop = null;

    document.getElementById('prediction-section').style.display = 'none';
    document.getElementById('evaluation-section').style.display = 'none';
    document.getElementById('stop-section').style.display = 'none';
    document.getElementById('trip-section').style.display = 'none';

    document.getElementById('routes-list').innerHTML = '';
    document.getElementById('trips-list').innerHTML = '';
    document.getElementById('stops-list').innerHTML = '';

    document.querySelectorAll('.item').forEach(el => {
        el.classList.remove('selected');
    });
}

// Event listeners
document.getElementById('load-routes-btn').addEventListener('click', loadRoutes);
document.getElementById('stop-session-btn').addEventListener('click', handleStopSession);
document.getElementById('new-session-btn').addEventListener('click', resetUI);

// Initial load
console.log('Live ETA UI loaded');
