import os
import django
import random
from datetime import datetime, timedelta

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'calibration.settings')
django.setup()

from core.models import JointType, CalibrationSession, SensorData, CalibrationModel

def create_test_data():
    # Create a joint type
    joint_type = JointType.objects.create(
        name='Test Joint Type',
        description='A test joint type for demonstration'
    )
    print(f'Created joint type: {joint_type.name}')

    # Create a calibration session
    session = CalibrationSession.objects.create(
        joint_type=joint_type,
        date_time=datetime.now(),
        notes='Test calibration session'
    )
    print(f'Created session: {session}')

    # Create some sensor data
    for i in range(10):
        sensor_data = SensorData.objects.create(
            session=session,
            applied_load=random.uniform(0, 100),
            sensor_voltage=random.uniform(0, 5),
            temperature=random.uniform(20, 30),
            humidity=random.uniform(40, 60),
            machine_speed=random.uniform(100, 500),
            material_type='Steel',
            true_output=random.uniform(0, 100)
        )
    print('Created 10 sensor data entries')

    # Create a calibration model
    model = CalibrationModel.objects.create(
        session=session,
        model_type='linear',
        model_file='test_model.pkl',
        accuracy=95.5,
        r2_score=0.98,
        mae=0.02,
        training_date=datetime.now()
    )
    print(f'Created model: {model}')

if __name__ == '__main__':
    create_test_data() 