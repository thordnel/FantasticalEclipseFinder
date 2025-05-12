eel.expose(updateProgress); // Expose JS function to Python
function updateProgress(message) {
    const progressDiv = document.getElementById('progress');
    progressDiv.innerHTML += message + "<br>"; // Append message
    if (message.startsWith("Progress") || message.startsWith("[")) { // A bit hacky, improve as needed
        progressDiv.scrollTop = progressDiv.scrollHeight; // Auto-scroll
    }
}

eel.expose(showSummary);
function showSummary(summaryText) {
    document.getElementById('output-summary').textContent = summaryText;
    document.getElementById('progress').innerHTML += "--- Calculation Complete --- <br>";
}

eel.expose(addHitDetail);
function addHitDetail(hitHtml) {
    document.getElementById('hits-details').innerHTML += hitHtml;
}


async function runCalculation() {
    document.getElementById('progress').innerHTML = "Starting calculation...<br>";
    document.getElementById('output-summary').textContent = "";
    document.getElementById('hits-details').innerHTML = "";

    let params = {
        lat1: parseFloat(document.getElementById('lat1').value),
        lon1: parseFloat(document.getElementById('lon1').value),
        alt1: parseFloat(document.getElementById('alt1').value),
        time1: document.getElementById('time1').value,
        lat2: parseFloat(document.getElementById('lat2').value),
        lon2: parseFloat(document.getElementById('lon2').value),
        alt2: parseFloat(document.getElementById('alt2').value),
        day2: parseInt(document.getElementById('day2').value),
        hms2: document.getElementById('hms2').value,
        start_year2: parseInt(document.getElementById('start_year2').value),
        end_year2: parseInt(document.getElementById('end_year2').value),
        months2_str: document.getElementById('months2').value, // Send as string
        search_hours_offset: parseFloat(document.getElementById('search_hours_offset').value)
    };

    // Call Python function
    // The Python function will call updateProgress and showSummary
    await eel.start_calculation(params)(); 
    // Note the extra () after the await eel.start_calculation(params)
    // This is because eel.start_calculation(params) itself returns a function that needs to be called.
}