from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.db.models import Avg, Count
from django.utils import timezone
from datetime import timedelta
import json
from .models import JointType, CalibrationSession, SensorData, CalibrationModel, HydraulicSystemData, WeldQualityModel, WeldInspection
from django.contrib import messages
from .kaggle_utils import download_dataset, load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import joblib
import os
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage

def dashboard(request):
    """Dashboard view showing overview of calibration data."""
    # Get recent sessions
    recent_sessions = CalibrationSession.objects.select_related('joint_type').order_by('-date_time')[:5]
    
    # Get statistics
    total_sessions = CalibrationSession.objects.count()
    total_models = CalibrationModel.objects.count()
    total_data_points = SensorData.objects.count()
    
    # Get active sessions (sessions from last 24 hours)
    active_sessions = CalibrationSession.objects.filter(
        date_time__gte=timezone.now() - timedelta(days=1)
    ).count()
    
    # Get model performance data for the line chart
    recent_models = CalibrationModel.objects.order_by('training_date')[:10]
    model_dates = [model.training_date.strftime('%Y-%m-%d') for model in recent_models]
    model_accuracies = [float(model.accuracy or 0) for model in recent_models]
    model_r2_scores = [float(model.r2_score or 0) for model in recent_models]
    
    # Get recent activities
    recent_activities = []
    
    # Add recent model trainings
    for model in CalibrationModel.objects.order_by('-training_date')[:5]:
        recent_activities.append({
            'date': model.training_date.strftime('%Y-%m-%d %H:%M'),
            'type': 'Model Training',
            'details': f'{model.model_type} model trained (Accuracy: {model.accuracy:.2f})'
        })
    
    # Add recent sessions
    for session in recent_sessions:
        recent_activities.append({
            'date': session.date_time.strftime('%Y-%m-%d %H:%M'),
            'type': 'Session Created',
            'details': f'New session for {session.joint_type.name}'
        })
    
    # Sort activities by date
    recent_activities.sort(key=lambda x: x['date'], reverse=True)
    recent_activities = recent_activities[:5]  # Keep only 5 most recent
    
    context = {
        'recent_sessions': recent_sessions,
        'total_sessions': total_sessions,
        'total_models': total_models,
        'total_data_points': total_data_points,
        'active_sessions': active_sessions,
        'model_dates': json.dumps(model_dates),
        'model_accuracies': json.dumps(model_accuracies),
        'model_r2_scores': json.dumps(model_r2_scores),
        'recent_activities': recent_activities,
    }
    return render(request, 'dashboard.html', context)

def joint_type_list(request):
    """List all joint types."""
    joint_types = JointType.objects.all()
    return render(request, 'joint_types/list.html', {'joint_types': joint_types})

def joint_type_detail(request, pk):
    """Show details of a specific joint type."""
    joint_type = get_object_or_404(JointType, pk=pk)
    sessions = joint_type.sessions.all().order_by('-date_time')
    return render(request, 'joint_types/detail.html', {
        'joint_type': joint_type,
        'sessions': sessions
    })

def joint_type_create(request):
    """Create a new joint type."""
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        JointType.objects.create(name=name, description=description)
        return redirect('joint_type_list')
    return render(request, 'joint_types/form.html')

def joint_type_update(request, pk):
    """Update an existing joint type."""
    joint_type = get_object_or_404(JointType, pk=pk)
    if request.method == 'POST':
        joint_type.name = request.POST.get('name')
        joint_type.description = request.POST.get('description')
        joint_type.save()
        return redirect('joint_type_detail', pk=pk)
    return render(request, 'joint_types/form.html', {'joint_type': joint_type})

def joint_type_delete(request, pk):
    """Delete a joint type."""
    joint_type = get_object_or_404(JointType, pk=pk)
    if request.method == 'POST':
        joint_type.delete()
        return redirect('joint_type_list')
    return render(request, 'joint_types/confirm_delete.html', {'joint_type': joint_type})

def session_list(request):
    """List all calibration sessions."""
    sessions = CalibrationSession.objects.select_related('joint_type').order_by('-date_time')
    return render(request, 'sessions/list.html', {'sessions': sessions})

def session_detail(request, pk):
    """Show details of a specific calibration session."""
    session = get_object_or_404(CalibrationSession, pk=pk)
    sensor_data = session.sensor_data.all().order_by('applied_load')
    models = session.models.all().order_by('-training_date')
    return render(request, 'sessions/detail.html', {
        'session': session,
        'sensor_data': sensor_data,
        'models': models
    })

def session_create(request):
    """Create a new calibration session."""
    if request.method == 'POST':
        joint_type_id = request.POST.get('joint_type')
        joint_type = get_object_or_404(JointType, pk=joint_type_id)
        CalibrationSession.objects.create(joint_type=joint_type)
        return redirect('session_list')
    joint_types = JointType.objects.all()
    return render(request, 'sessions/form.html', {'joint_types': joint_types})

def session_update(request, pk):
    """Update an existing calibration session."""
    session = get_object_or_404(CalibrationSession, pk=pk)
    if request.method == 'POST':
        session.joint_type_id = request.POST.get('joint_type')
        session.notes = request.POST.get('notes')
        session.save()
        return redirect('session_detail', pk=pk)
    joint_types = JointType.objects.all()
    return render(request, 'sessions/form.html', {'session': session, 'joint_types': joint_types})

def session_delete(request, pk):
    """Delete a calibration session."""
    session = get_object_or_404(CalibrationSession, pk=pk)
    if request.method == 'POST':
        session.delete()
        return redirect('session_list')
    return render(request, 'sessions/confirm_delete.html', {'session': session})

def sensor_data_list(request):
    """List all sensor data entries."""
    # Get filter parameters
    session_id = request.GET.get('session')
    material_type = request.GET.get('material_type')
    
    # Base queryset
    queryset = SensorData.objects.select_related('session', 'session__joint_type')
    
    # Apply filters
    if session_id:
        queryset = queryset.filter(session_id=session_id)
    if material_type:
        queryset = queryset.filter(material_type=material_type)
    
    # Order by creation date
    data = queryset.order_by('-created_at')
    
    # Get all sessions for filter dropdown
    sessions = CalibrationSession.objects.select_related('joint_type').order_by('-date_time')
    
    # Get unique material types for filter dropdown
    material_types = SensorData.objects.values_list('material_type', flat=True).distinct()
    
    context = {
        'data': data,
        'sessions': sessions,
        'material_types': material_types,
        'selected_session': session_id,
        'selected_material': material_type,
    }
    return render(request, 'sensor_data/list.html', context)

def sensor_data_detail(request, pk):
    """Show details of a specific sensor data entry."""
    data = get_object_or_404(SensorData, pk=pk)
    return render(request, 'sensor_data/detail.html', {'data': data})

def sensor_data_create(request):
    """Create a new sensor data entry."""
    if request.method == 'POST':
        session_id = request.POST.get('session')
        session = get_object_or_404(CalibrationSession, pk=session_id)
        SensorData.objects.create(
            session=session,
            applied_load=request.POST.get('applied_load'),
            sensor_voltage=request.POST.get('sensor_voltage'),
            temperature=request.POST.get('temperature'),
            humidity=request.POST.get('humidity'),
            machine_speed=request.POST.get('machine_speed'),
            material_type=request.POST.get('material_type'),
            true_output=request.POST.get('true_output')
        )
        return redirect('sensor_data_list')
    sessions = CalibrationSession.objects.all()
    return render(request, 'sensor_data/form.html', {'sessions': sessions})

def sensor_data_update(request, pk):
    """Update an existing sensor data entry."""
    data = get_object_or_404(SensorData, pk=pk)
    if request.method == 'POST':
        session_id = request.POST.get('session')
        session = get_object_or_404(CalibrationSession, pk=session_id)
        data.session = session
        data.applied_load = request.POST.get('applied_load')
        data.sensor_voltage = request.POST.get('sensor_voltage')
        data.temperature = request.POST.get('temperature')
        data.humidity = request.POST.get('humidity')
        data.machine_speed = request.POST.get('machine_speed')
        data.material_type = request.POST.get('material_type')
        data.true_output = request.POST.get('true_output')
        data.save()
        return redirect('sensor_data_detail', pk=pk)
    sessions = CalibrationSession.objects.all()
    return render(request, 'sensor_data/form.html', {'data': data, 'sessions': sessions})

def sensor_data_delete(request, pk):
    """Delete a sensor data entry."""
    data = get_object_or_404(SensorData, pk=pk)
    if request.method == 'POST':
        data.delete()
        return redirect('sensor_data_list')
    return render(request, 'sensor_data/confirm_delete.html', {'data': data})

def model_list(request):
    """List all calibration models."""
    models = CalibrationModel.objects.select_related('session').order_by('-training_date')
    return render(request, 'models/list.html', {'models': models})

def model_detail(request, pk):
    """Show details of a specific calibration model."""
    model = get_object_or_404(CalibrationModel, pk=pk)
    return render(request, 'models/detail.html', {'model': model})

def model_create(request):
    """Create a new calibration model."""
    if request.method == 'POST':
        session_id = request.POST.get('session')
        session = get_object_or_404(CalibrationSession, pk=session_id)
        model_file = request.FILES.get('model_file')
        CalibrationModel.objects.create(
            session=session,
            model_type=request.POST.get('model_type'),
            model_file=model_file,
            accuracy=request.POST.get('accuracy'),
            r2_score=request.POST.get('r2_score'),
            mae=request.POST.get('mae')
        )
        return redirect('model_list')
    sessions = CalibrationSession.objects.all()
    return render(request, 'models/form.html', {'sessions': sessions})

def model_update(request, pk):
    """Update an existing calibration model."""
    model = get_object_or_404(CalibrationModel, pk=pk)
    if request.method == 'POST':
        session_id = request.POST.get('session')
        session = get_object_or_404(CalibrationSession, pk=session_id)
        model.session = session
        model.model_type = request.POST.get('model_type')
        if 'model_file' in request.FILES:
            model.model_file = request.FILES['model_file']
        model.accuracy = request.POST.get('accuracy')
        model.r2_score = request.POST.get('r2_score')
        model.mae = request.POST.get('mae')
        model.save()
        return redirect('model_detail', pk=pk)
    sessions = CalibrationSession.objects.all()
    return render(request, 'models/form.html', {'model': model, 'sessions': sessions})

def model_delete(request, pk):
    """Delete a calibration model."""
    model = get_object_or_404(CalibrationModel, pk=pk)
    if request.method == 'POST':
        model.delete()
        return redirect('model_list')
    return render(request, 'models/confirm_delete.html', {'model': model})

@require_http_methods(["POST"])
def model_predict(request, pk):
    """Make predictions using a calibration model."""
    model = get_object_or_404(CalibrationModel, pk=pk)
    try:
        data = json.loads(request.body)
        features = np.array([
            float(data['sensor_voltage']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['machine_speed'])
        ]).reshape(1, -1)
        
        # Load the model
        model_file = joblib.load(model.model_file.path)
        
        # Handle KRR models differently
        if model.model_type == 'krr':
            loaded_model = model_file['model']
            scaler = model_file['scaler']
            features_scaled = scaler.transform(features)
            prediction = float(loaded_model.predict(features_scaled)[0])
        else:
            loaded_model = model_file
            prediction = float(loaded_model.predict(features)[0])
            
        return JsonResponse({'prediction': prediction})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
def download_kaggle_data(request):
    try:
        # Create a new session
        session = CalibrationSession.objects.create(
            operator=request.user,
            session_type='temperature',
            description='Temperature sensor calibration using IOT-temp dataset'
        )
        
        # Read the temperature dataset
        df = pd.read_csv('IOT-temp.csv')
        
        # Preprocess the data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Create features and target
        X = df[['hour', 'day_of_week', 'temperature']]
        y = df['temperature']  # Using temperature as both feature and target for demonstration
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a scaler for KRR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Kernel Ridge': KernelRidge(kernel='rbf', alpha=0.1)
        }
        
        for model_name, model in models.items():
            # Train the model
            if model_name == 'Kernel Ridge':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Save the model
            model_path = f'media/models/{session.id}_{model_name.lower().replace(" ", "_")}.pkl'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # For KRR, save both the model and scaler
            if model_name == 'Kernel Ridge':
                joblib.dump({'model': model, 'scaler': scaler}, model_path)
            else:
                joblib.dump(model, model_path)
            
            # Create model record
            model_type = 'krr' if model_name == 'Kernel Ridge' else model_name.lower().replace(' ', '_')
            CalibrationModel.objects.create(
                session=session,
                model_type=model_type,
                model_file=model_path,
                accuracy=float(r2),
                r2_score=float(r2),
                mae=float(mae),
                training_date=timezone.now()
            )
            
            # Create sensor data entries
            for _, row in df.iterrows():
                SensorData.objects.create(
                    session=session,
                    applied_load=0.0,  # Not applicable for temperature data
                    sensor_voltage=row['temperature'],  # Using temperature as voltage
                    temperature=row['temperature'],
                    humidity=0.0,  # Not available in dataset
                    machine_speed=0.0,  # Not applicable
                    material_type='temperature',  # Using 'temperature' as material type
                    true_output=row['temperature']  # Using temperature as true output
                )
        
        messages.success(request, 'Temperature data imported and models trained successfully!')
        return redirect('session_detail', session_id=session.id)
        
    except Exception as e:
        messages.error(request, f'Error importing data: {str(e)}')
        return redirect('session_list')

def hydraulic_data_list(request):
    """List all hydraulic system data entries."""
    data = HydraulicSystemData.objects.all().order_by('-timestamp')
    return render(request, 'hydraulic/list.html', {'data': data})

def hydraulic_data_detail(request, pk):
    """Show details of a specific hydraulic system data entry."""
    data = get_object_or_404(HydraulicSystemData, pk=pk)
    return render(request, 'hydraulic/detail.html', {'data': data})

def import_hydraulic_data(request):
    """Import hydraulic system data from the dataset."""
    if request.method == 'POST':
        try:
            # Read profile.txt for condition indicators
            profile_path = 'templates/joint_types/condition+monitoring+of+hydraulic+systems/profile.txt'
            with open(profile_path, 'r') as f:
                profiles = [line.strip().split() for line in f.readlines()]
            
            # Process each data point
            for i, profile in enumerate(profiles):
                # Create HydraulicSystemData instance
                hydraulic_data = HydraulicSystemData(
                    cooler_condition=float(profile[0]),
                    valve_condition=float(profile[1]),
                    pump_leakage=int(profile[2]),
                    accumulator_pressure=float(profile[3]),
                    is_stable=bool(int(profile[4]))
                )
                
                # Read and process sensor data files
                sensor_data = {}
                
                # Pressure sensors (PS1-6)
                for j in range(1, 7):
                    with open(f'templates/joint_types/condition+monitoring+of+hydraulic+systems/PS{j}.txt', 'r') as f:
                        sensor_data[f'PS{j}'] = [float(x) for x in f.readlines()[i].strip().split()]
                
                # Motor power (EPS1)
                with open('templates/joint_types/condition+monitoring+of+hydraulic+systems/EPS1.txt', 'r') as f:
                    sensor_data['EPS1'] = [float(x) for x in f.readlines()[i].strip().split()]
                
                # Volume flow (FS1/2)
                for j in range(1, 3):
                    with open(f'templates/joint_types/condition+monitoring+of+hydraulic+systems/FS{j}.txt', 'r') as f:
                        sensor_data[f'FS{j}'] = [float(x) for x in f.readlines()[i].strip().split()]
                
                # Temperature (TS1-4)
                for j in range(1, 5):
                    with open(f'templates/joint_types/condition+monitoring+of+hydraulic+systems/TS{j}.txt', 'r') as f:
                        sensor_data[f'TS{j}'] = [float(x) for x in f.readlines()[i].strip().split()]
                
                # Vibration (VS1)
                with open('templates/joint_types/condition+monitoring+of+hydraulic+systems/VS1.txt', 'r') as f:
                    sensor_data['VS1'] = [float(x) for x in f.readlines()[i].strip().split()]
                
                # Efficiency (SE)
                with open('templates/joint_types/condition+monitoring+of+hydraulic+systems/SE.txt', 'r') as f:
                    sensor_data['SE'] = [float(x) for x in f.readlines()[i].strip().split()]
                
                # Cooling efficiency (CE)
                with open('templates/joint_types/condition+monitoring+of+hydraulic+systems/CE.txt', 'r') as f:
                    sensor_data['CE'] = [float(x) for x in f.readlines()[i].strip().split()]
                
                # Cooling power (CP)
                with open('templates/joint_types/condition+monitoring+of+hydraulic+systems/CP.txt', 'r') as f:
                    sensor_data['CP'] = [float(x) for x in f.readlines()[i].strip().split()]
                
                # Save sensor data to JSON fields
                hydraulic_data.pressure_sensors = {k: v for k, v in sensor_data.items() if k.startswith('PS')}
                hydraulic_data.motor_power = sensor_data['EPS1']
                hydraulic_data.volume_flow = {k: v for k, v in sensor_data.items() if k.startswith('FS')}
                hydraulic_data.temperature = {k: v for k, v in sensor_data.items() if k.startswith('TS')}
                hydraulic_data.vibration = sensor_data['VS1']
                hydraulic_data.efficiency = sensor_data['SE']
                hydraulic_data.cooling_efficiency = sensor_data['CE']
                hydraulic_data.cooling_power = sensor_data['CP']
                
                hydraulic_data.save()
            
            messages.success(request, 'Hydraulic system data imported successfully!')
            return redirect('hydraulic_data_list')
            
        except Exception as e:
            messages.error(request, f'Error importing hydraulic system data: {str(e)}')
            return redirect('hydraulic_data_list')
    
    return render(request, 'hydraulic/import.html')

def visualization(request):
    """Render the visualization page with graphs."""
    return render(request, 'visualization/graphs.html')

def train_weld_model(request):
    if request.method == 'POST':
        # For now, just create a dummy model
        model_path = 'media/weld_models/dummy_model.h5'
        with open(model_path, 'w') as f:
            f.write('Dummy model file')
        
        weld_model = WeldQualityModel.objects.create(
            name='Dummy Weld Model',
            model_file=model_path,
            accuracy=0.85,
            classes=['bad_welding', 'crack', 'excess_reinforcement', 'good_welding', 'porosity', 'spatters'],
            input_shape=[224, 224, 3]
        )
        
        return redirect('weld_model_detail', pk=weld_model.pk)
    
    return render(request, 'weld/train.html')

def weld_model_detail(request, pk):
    model = get_object_or_404(WeldQualityModel, pk=pk)
    return render(request, 'weld/detail.html', {'model': model})

def predict_weld_quality(request, pk):
    if request.method == 'POST':
        model = get_object_or_404(WeldQualityModel, pk=pk)
        image_file = request.FILES['image']
        
        # For now, just create a dummy prediction
        import random
        prediction = random.choice(model.classes)
        confidence = random.uniform(0.8, 0.95)
        
        # Save inspection result
        inspection = WeldInspection.objects.create(
            image=image_file,
            prediction=prediction,
            confidence=confidence,
            model_used=model
        )
        
        return render(request, 'weld/result.html', {
            'inspection': inspection,
            'confidence': confidence * 100
        })
    
    return render(request, 'weld/predict.html', {'model_id': pk})

def analyze_data(request):
    """Perform comprehensive data analysis."""
    try:
        # Read the dataset
        df = pd.read_csv('IOT-temp.csv')
        
        # Basic EDA
        total_samples = len(df)
        missing_values = df.isnull().sum()
        basic_stats = df.describe()
        
        # Time series analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Create features and target
        X = df[['hour', 'day_of_week', 'temperature']]
        y = df['temperature']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train KRR model
        krr = KernelRidge(kernel='rbf', alpha=0.1)
        krr.fit(X_train_scaled, y_train)
        y_pred = krr.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Create visualizations
        plt.switch_backend('Agg')
        
        # 1. Temperature Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='temperature', bins=30)
        plt.title('Temperature Distribution')
        temp_dist_img = get_plot_image()
        
        # 2. Temperature by Hour
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='hour', y='temperature')
        plt.title('Temperature Distribution by Hour')
        temp_hour_img = get_plot_image()
        
        # 3. Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Temperature')
        plt.ylabel('Predicted Temperature')
        plt.title('Actual vs Predicted Temperature')
        pred_img = get_plot_image()
        
        # 4. Residuals Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Temperature')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        residuals_img = get_plot_image()
        
        context = {
            'total_samples': total_samples,
            'missing_values': missing_values.to_dict(),
            'basic_stats': basic_stats.to_html(classes='table table-striped'),
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'temp_dist_img': temp_dist_img,
            'temp_hour_img': temp_hour_img,
            'pred_img': pred_img,
            'residuals_img': residuals_img,
        }
        
        return render(request, 'analysis/dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f'Error performing analysis: {str(e)}')
        return redirect('dashboard')

def get_plot_image():
    """Convert matplotlib plot to base64 image."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    graphic = base64.b64encode(image_png)
    return graphic.decode('utf-8')