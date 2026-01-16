// Load data and initialize dashboard
document.addEventListener('DOMContentLoaded', async () => {
    await loadData();
    initializeCharts();
    setupNavigation();
});

let predictionData = null;
let comparisonData = null;
let dmTestData = null;
let regimeData = null;

// Load CSV data
async function loadData() {
    try {
        // Load predictions
        const predResp = await fetch('../results/predictions/all_predictions.csv');
        const predText = await predResp.text();
        predictionData = parseCSV(predText);

        // Load comparison
        const compResp = await fetch('../results/tables/model_comparison.csv');
        const compText = await compResp.text();
        comparisonData = parseCSV(compText);

        // Load DM tests
        const dmResp = await fetch('../results/tables/dm_test_results.csv');
        const dmText = await dmResp.text();
        dmTestData = parseCSV(dmText);

        // Load regime analysis
        const regimeResp = await fetch('../results/tables/regime_analysis.csv');
        const regimeText = await regimeResp.text();
        regimeData = parseCSV(regimeText);

        console.log('✓ All data loaded successfully');
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

// CSV Parser
function parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const row = {};
        headers.forEach((header, index) => {
            row[header] = values[index] ? values[index].trim() : '';
        });
        data.push(row);
    }

    return data;
}

// Initialize all charts
function initializeCharts() {
    createPredictionsChart();
    createMSEChart();
    createMAEChart();
    createAccuracyChart();
    createErrorDistChart();
    createRegimeHeatmap();
    populateDMTable();
}

// Predictions Time Series Chart (Plotly)
function createPredictionsChart() {
    if (!predictionData) return;

    // Sample data for faster rendering (every 5th point)
    const sampledData = predictionData.filter((_, i) => i % 5 === 0);

    const dates = sampledData.map(row => row.Date);
    const actual = sampledData.map(row => parseFloat(row.Actual));
    const garch = sampledData.map(row => parseFloat(row.GARCH_Volatility));
    const lstm = sampledData.map(row => parseFloat(row.LSTM_Prediction));
    const hybrid = sampledData.map(row => parseFloat(row.Hybrid_Prediction));

    const trace1 = {
        x: dates,
        y: actual,
        name: 'Actual',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#1f2937', width: 2 }
    };

    const trace2 = {
        x: dates,
        y: lstm,
        name: 'LSTM',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#3b82f6', width: 2, dash: 'dash' }
    };

    const trace3 = {
        x: dates,
        y: hybrid,
        name: 'Hybrid GARCH-LSTM',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#f5576c', width: 2.5 }
    };

    const layout = {
        title: {
            text: 'EUR/USD Exchange Rate Predictions',
            font: { size: 20, weight: 700 }
        },
        xaxis: { 
            title: 'Date',
            showgrid: true,
            gridcolor: '#e5e7eb'
        },
        yaxis: { 
            title: 'Exchange Rate',
            showgrid: true,
            gridcolor: '#e5e7eb'
        },
        hovermode: 'x unified',
        legend: {
            orientation: 'h',
            y: -0.15,
            x: 0.5,
            xanchor: 'center'
        },
        height: 500,
        margin: { t: 60, b: 80, l: 60, r: 40 }
    };

    Plotly.newPlot('predictionsChart', [trace1, trace2, trace3], layout, {
        responsive: true,
        displayModeBar: true
    });
}

// MSE Comparison Chart (Chart.js)
function createMSEChart() {
    const ctx = document.getElementById('mseChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['GARCH(1,1)', 'LSTM Baseline', 'Hybrid GARCH-LSTM'],
            datasets: [{
                label: 'MSE',
                data: [0.000024, 0.000004, 0.000002],
                backgroundColor: [
                    'rgba(251, 191, 36, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(245, 87, 108, 0.8)'
                ],
                borderColor: [
                    'rgba(251, 191, 36, 1)',
                    'rgba(59, 130, 246, 1)',
                    'rgba(245, 87, 108, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => `MSE: ${context.parsed.y.toFixed(6)}`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#e5e7eb' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

// MAE Comparison Chart
function createMAEChart() {
    const ctx = document.getElementById('maeChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['GARCH(1,1)', 'LSTM Baseline', 'Hybrid GARCH-LSTM'],
            datasets: [{
                label: 'MAE',
                data: [0.003543, 0.001634, 0.000945],
                backgroundColor: [
                    'rgba(251, 191, 36, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(245, 87, 108, 0.8)'
                ],
                borderColor: [
                    'rgba(251, 191, 36, 1)',
                    'rgba(59, 130, 246, 1)',
                    'rgba(245, 87, 108, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => `MAE: ${context.parsed.y.toFixed(6)}`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#e5e7eb' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

// Directional Accuracy Chart
function createAccuracyChart() {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['GARCH(1,1): 51.6%', 'LSTM: 77.78%', 'Hybrid: 86.20%'],
            datasets: [{
                data: [51.60, 77.78, 86.20],
                backgroundColor: [
                    'rgba(251, 191, 36, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(245, 87, 108, 0.8)'
                ],
                borderColor: [
                    'rgba(251, 191, 36, 1)',
                    'rgba(59, 130, 246, 1)',
                    'rgba(245, 87, 108, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        font: { size: 12, weight: '600' }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.label}: ${context.parsed.toFixed(2)}%`
                    }
                }
            }
        }
    });
}

// Error Distribution Chart
function createErrorDistChart() {
    if (!predictionData) return;

    const ctx = document.getElementById('errorDistChart').getContext('2d');
    
    // Calculate errors
    const lstmErrors = predictionData.map(row => 
        parseFloat(row.LSTM_Error)
    );
    const hybridErrors = predictionData.map(row => 
        parseFloat(row.Hybrid_Error)
    );

    // Create histogram bins
    const bins = 20;
    const lstmHist = createHistogram(lstmErrors, bins);
    const hybridHist = createHistogram(hybridErrors, bins);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: lstmHist.labels,
            datasets: [
                {
                    label: 'LSTM Errors',
                    data: lstmHist.counts,
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Hybrid Errors',
                    data: hybridHist.counts,
                    backgroundColor: 'rgba(245, 87, 108, 0.6)',
                    borderColor: 'rgba(245, 87, 108, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Frequency' },
                    grid: { color: '#e5e7eb' }
                },
                x: {
                    title: { display: true, text: 'Prediction Error' },
                    grid: { display: false }
                }
            }
        }
    });
}

// Helper function to create histogram
function createHistogram(data, bins) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / bins;
    
    const counts = new Array(bins).fill(0);
    const labels = [];

    for (let i = 0; i < bins; i++) {
        const binStart = min + i * binWidth;
        const binEnd = binStart + binWidth;
        labels.push(binStart.toFixed(4));
        
        data.forEach(value => {
            if (value >= binStart && value < binEnd) {
                counts[i]++;
            }
        });
    }

    return { labels, counts };
}

// Regime Performance Heatmap (Plotly)
function createRegimeHeatmap() {
    if (!regimeData) return;

    const regimes = regimeData.map(row => row.Volatility_Regime);
    const garchMSE = regimeData.map(row => parseFloat(row.GARCH_MSE) * 1000000); // Scale for visibility
    const lstmMSE = regimeData.map(row => parseFloat(row.LSTM_MSE) * 1000000);
    const hybridMSE = regimeData.map(row => parseFloat(row.Hybrid_MSE) * 1000000);

    const trace = {
        z: [garchMSE, lstmMSE, hybridMSE],
        x: regimes,
        y: ['GARCH(1,1)', 'LSTM', 'Hybrid'],
        type: 'heatmap',
        colorscale: [
            [0, '#10b981'],
            [0.5, '#fbbf24'],
            [1, '#ef4444']
        ],
        showscale: true,
        hovertemplate: 'Model: %{y}<br>Regime: %{x}<br>MSE (×10⁻⁶): %{z:.4f}<extra></extra>'
    };

    const layout = {
        title: {
            text: 'Model Performance Across Volatility Regimes',
            font: { size: 18, weight: 700 }
        },
        xaxis: { title: 'Volatility Regime', side: 'bottom' },
        yaxis: { title: 'Model' },
        height: 400,
        margin: { t: 60, b: 80, l: 100, r: 40 }
    };

    Plotly.newPlot('regimeHeatmap', [trace], layout, { responsive: true });
}

// Populate DM Test Table
function populateDMTable() {
    if (!dmTestData) return;

    const tbody = document.getElementById('dmTestTable');
    tbody.innerHTML = '';

    dmTestData.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.Comparison}</td>
            <td>${parseFloat(row.DM_Statistic).toFixed(4)}</td>
            <td>${parseFloat(row.p_value).toExponential(2)}</td>
            <td><span class="winner-badge">${row.Winner}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

// Navigation Smooth Scroll & Active State
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            
            // Add active class to clicked link
            link.classList.add('active');
            
            // Smooth scroll to section
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // Update active link on scroll
    window.addEventListener('scroll', () => {
        let current = '';
        const sections = document.querySelectorAll('.section, .hero');
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}
