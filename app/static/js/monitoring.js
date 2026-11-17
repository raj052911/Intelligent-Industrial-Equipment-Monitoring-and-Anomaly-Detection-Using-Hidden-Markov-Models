
document.addEventListener("DOMContentLoaded", () => {
    loadMonitoringData();
});

async function loadMonitoringData() {
    try {
        const response = await fetch("/api/sensor-data");

        if (!response.ok) {
            throw new Error("Unable to load monitoring data");
        }

        const data = await response.json();

        updateMonitoringTable(data);
        updateMonitoringChart(data);

    } catch (error) {
        console.error("Monitoring error:", error);
    }
}

function updateMonitoringTable(data) {
    const tableBody = document.getElementById("sensorTableBody");

    if (!tableBody) return;

    tableBody.innerHTML = "";

    data.slice(-100).reverse().forEach(item => {
        const row = document.createElement("tr");

        const status = Number(item.predicted_anomaly) === 1
            ? "Anomaly"
            : "Normal";

        row.innerHTML = `
            <td>${item.timestamp ?? "-"}</td>
            <td>${Number(item.temperature).toFixed(2)}</td>
            <td>${Number(item.vibration).toFixed(3)}</td>
            <td>${Number(item.pressure).toFixed(2)}</td>
            <td>${status}</td>
        `;

        tableBody.appendChild(row);
    });
}

function updateMonitoringChart(data) {
    const canvas = document.getElementById("monitoringChart");

    if (!canvas) return;

    const labels = data.map((item, index) =>
        item.timestamp || index
    );

    const temperature = data.map(item => item.temperature);

    if (window.monitoringChartInstance) {
        window.monitoringChartInstance.destroy();
    }

    window.monitoringChartInstance = new Chart(canvas, {
        type: "line",

        data: {
            labels: labels,

            datasets: [{
                label: "Temperature",
                data: temperature,
                borderWidth: 1.5,
                pointRadius: 0
            }]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}
