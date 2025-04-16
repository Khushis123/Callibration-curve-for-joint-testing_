# Calibration Management System

A Django-based web application for managing calibration data and models, with a focus on Kernel Ridge Regression (KRR) implementation.

## Features

- Dashboard with real-time statistics and visualizations
- Joint type management
- Calibration session tracking
- Sensor data collection and analysis
- Multiple model types support:
  - Linear Regression
  - Polynomial Regression
  - Random Forest
  - Neural Network
  - Kernel Ridge Regression (KRR)
- Data visualization and analysis
- Hydraulic system monitoring
- Weld quality inspection

## Technology Stack

- Python 3.x
- Django 5.0.2
- Django REST Framework
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Bootstrap 5

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Khushis123/Callibration-curve-for-join-testing_.git
cd Callibration-curve-for-join-testing_
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run migrations:
```bash
python manage.py migrate
```

4. Start the development server:
```bash
python manage.py runserver
```

## Usage

1. Access the application at http://127.0.0.1:8000/
2. Navigate through different sections:
   - Joint Types
   - Sessions
   - Sensor Data
   - Models
   - Visualization
   - Data Analysis

## Author

- Khushi Arya
- Email: khushis12072005@gmail.com
- GitHub: [@Khushis123](https://github.com/Khushis123)

## Project Structure

```
calibration/
├── core/                 # Main application
│   ├── models.py        # Database models
│   ├── views.py         # View functions
│   ├── urls.py          # URL patterns
│   └── admin.py         # Admin configuration
├── templates/           # HTML templates
├── static/             # Static files
├── media/              # Uploaded files
└── manage.py           # Django management script
```

## API Endpoints

- `/api/sessions/` - Calibration sessions
- `/api/sensor-data/` - Sensor data
- `/api/models/` - Machine learning models
- `/api/hydraulic/` - Hydraulic system data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any queries or support, please contact [your-email@example.com] 