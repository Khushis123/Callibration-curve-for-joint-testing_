// Sensor Data JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const filterForm = document.getElementById('filter-form');
    const sensorDataTable = document.getElementById('sensor-data-table');

    if (filterForm) {
        filterForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(filterForm);
            const sessionId = formData.get('session');
            const materialType = formData.get('material_type');

            // Build query string
            let queryString = '?';
            if (sessionId) queryString += `session=${sessionId}&`;
            if (materialType) queryString += `material_type=${materialType}`;

            // Redirect to filtered URL
            window.location.href = `/sensor-data/${queryString}`;
        });
    }

    // Add hover effect to table rows
    if (sensorDataTable) {
        const rows = sensorDataTable.getElementsByTagName('tr');
        for (let i = 1; i < rows.length; i++) {
            rows[i].addEventListener('mouseenter', function() {
                this.style.backgroundColor = '#f8f9fa';
                this.style.transition = 'background-color 0.3s ease';
            });
            
            rows[i].addEventListener('mouseleave', function() {
                this.style.backgroundColor = '';
            });
        }
    }

    // Add animation to stats cards
    const statsCards = document.querySelectorAll('.stats-card');
    statsCards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 200);
    });
}); 