// web/script.js

// Expose functions to Python
eel.expose(updateProgress);
eel.expose(updateProgressBar); // NEW
eel.expose(addHitDetail);
eel.expose(showSummary);

const logArea = document.getElementById('log-area');
const resultsArea = document.getElementById('results-area');
const progressBar = document.getElementById('progress-bar');
const btn = document.getElementById('calc-btn');

function updateProgress(message) {
    logArea.innerText = `> ${message}`;
    logArea.scrollTop = logArea.scrollHeight;
}

// NEW: Python calls this with percentage (0-100)
function updateProgressBar(percent) {
    progressBar.style.width = percent + '%';
}

function addHitDetail(htmlContent) {
    const div = document.createElement('div');
    div.innerHTML = htmlContent;
    resultsArea.prepend(div);
}

function showSummary(message) {
    updateProgress(message);
    btn.disabled = false;
    btn.innerText = "Run Calculation";
}

async function runPythonCalc() {
    // Reset UI
    btn.disabled = true;
    btn.innerText = "Scanning...";
    resultsArea.innerHTML = '';
    progressBar.style.width = '0%'; // Reset bar

    // Gather Inputs
    const params = {
        ref_date_str: document.getElementById('ref_date').value,
        lat1: parseFloat(document.getElementById('lat1').value),
        lon1: parseFloat(document.getElementById('lon1').value),
        lat2: parseFloat(document.getElementById('lat2').value),
        lon2: parseFloat(document.getElementById('lon2').value),
        time1: document.getElementById('time1').value,
        search_hours_offset: parseInt(document.getElementById('search_hours_offset').value),
        search_radius: parseFloat(document.getElementById('search_radius').value),
        
        // NEW: Send Dynamic Years
        start_year: parseInt(document.getElementById('start_year').value),
        end_year: parseInt(document.getElementById('end_year').value)
    };

    try {
        await eel.start_calculation(params)();
    } catch (e) {
        updateProgress("Error: " + e);
        btn.disabled = false;
    }
}