document.addEventListener("DOMContentLoaded", () => {
    loadDashboardMetrics();
    loadSensorChart();
    loadModelInformation();
});

async function loadDashboardMetrics() {
    try {
        const response = await fetch("/api/metrics");

        if (!response.ok) {
            throw new Error("Unable to load dashboard metrics");
        }

        const data = await response.json();

        updateElement("total-readings", data.total_readings);
        updateElement("normal-readings", data.normal_readings);
        updateElement("anomaly-count", data.anomalies);
        updateElement("anomaly-rate", data.anomaly_rate + "%");

    } catch (error) {
        console.error("Metrics error:", error);
    }
}

async function loadModelInformation() {
    try {
        const response = await fetch("/api/model-info");

        if (!response.ok) {
            throw new Error("Unable to load model information");
        }

        const data = await response.json();

        updateElement("model-states", data.hidden_states);
        updateElement("model-status", data.model_status);

    } catch (error) {
        console.error("Model information error:", error);
    }
}

async function loadSensorChart() {
    try {
        const response = await fetch("/api/sensor-data");

        if (!response.ok) {
            throw new Error("Unable to load sensor data");
        }

        const data = await response.json();

        const labels = data.map((item, index) =>
            item.timestamp || index
        );

        const temperature = data.map(item => item.temperature);
        const vibration = data.map(item => item.vibration);
        const pressure = data.map(item => item.pressure);

        const anomalies = data.map(item =>
            item.predicted_anomaly === 1
                ? item.temperature
                : null
        );

        const canvas = document.getElementById("sensorChart");

        if (!canvas) return;

        if (window.sensorChartInstance) {
            window.sensorChartInstance.destroy();
        }

        window.sensorChartInstance = new Chart(canvas, {
            type: "line",

            data: {
                labels: labels,

                datasets: [
                    {
                        label: "Temperature",
                        data: temperature,
                        borderWidth: 1.5,
                        pointRadius: 0
                    },
                    {
                        label: "Vibration",
                        data: vibration,
                        borderWidth: 1.5,
                        pointRadius: 0
                    },
                    {
                        label: "Pressure",
                        data: pressure,
                        borderWidth: 1.5,
                        pointRadius: 0
                    },
                    {
                        label: "Detected Anomalies",
                        data: anomalies,
                        borderWidth: 0,
                        pointRadius: 3,
                        showLine: false
                    }
                ]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                interaction: {
                    intersect: false,
                    mode: "index"
                },

                scales: {
                    x: {
                        ticks: {
                            maxTicksLimit: 10
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error("Sensor chart error:", error);
    }
}

function updateElement(id, value) {
    const element = document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}
