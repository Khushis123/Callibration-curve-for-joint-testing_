from django.db import models
from django.utils import timezone

class JointType(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']

class CalibrationSession(models.Model):
    joint_type = models.ForeignKey(JointType, on_delete=models.CASCADE, related_name='sessions')
    date_time = models.DateTimeField(default=timezone.now)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.joint_type.name} - {self.date_time.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        ordering = ['-date_time']

class SensorData(models.Model):
    session = models.ForeignKey(CalibrationSession, on_delete=models.CASCADE, related_name='sensor_data')
    applied_load = models.FloatField()
    sensor_voltage = models.FloatField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    machine_speed = models.FloatField()
    material_type = models.CharField(max_length=50)
    true_output = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.session} - Load: {self.applied_load}"

    class Meta:
        ordering = ['session', 'applied_load']

class CalibrationModel(models.Model):
    MODEL_TYPES = [
        ('linear', 'Linear Regression'),
        ('polynomial', 'Polynomial Regression'),
        ('random_forest', 'Random Forest'),
        ('neural_network', 'Neural Network'),
        ('krr', 'Kernel Ridge Regression'),
    ]

    session = models.ForeignKey(CalibrationSession, on_delete=models.CASCADE, related_name='models')
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    model_file = models.FileField(upload_to='models/')
    accuracy = models.FloatField()
    r2_score = models.FloatField()
    mae = models.FloatField()
    training_date = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.session} - {self.get_model_type_display()}"

    class Meta:
        ordering = ['-training_date']

class HydraulicSystemData(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    pressure_sensors = models.JSONField()  # PS1-6
    motor_power = models.JSONField()  # EPS1
    volume_flow = models.JSONField()  # FS1/2
    temperature = models.JSONField()  # TS1-4
    vibration = models.JSONField()  # VS1
    efficiency = models.JSONField()  # SE
    cooling_efficiency = models.JSONField()  # CE
    cooling_power = models.JSONField()  # CP
    
    # Condition indicators
    cooler_condition = models.FloatField()  # %
    valve_condition = models.FloatField()  # %
    pump_leakage = models.IntegerField()  # 0, 1, 2
    accumulator_pressure = models.FloatField()  # bar
    is_stable = models.BooleanField()  # stable flag

    class Meta:
        verbose_name = "Hydraulic System Data"
        verbose_name_plural = "Hydraulic System Data"

    def __str__(self):
        return f"Hydraulic Data - {self.timestamp}"

class WeldQualityModel(models.Model):
    QUALITY_CHOICES = [
        ('bad_welding', 'Bad Welding'),
        ('crack', 'Crack'),
        ('excess_reinforcement', 'Excess Reinforcement'),
        ('good_welding', 'Good Welding'),
        ('porosity', 'Porosity'),
        ('spatters', 'Spatters'),
    ]
    
    name = models.CharField(max_length=100)
    model_file = models.FileField(upload_to='weld_models/')
    accuracy = models.FloatField(null=True, blank=True)
    training_date = models.DateTimeField(auto_now_add=True)
    classes = models.JSONField(default=list)
    input_shape = models.JSONField(default=list)
    
    def __str__(self):
        return f"{self.name} ({self.training_date.strftime('%Y-%m-%d')})"

class WeldInspection(models.Model):
    image = models.ImageField(upload_to='weld_inspections/')
    prediction = models.CharField(max_length=50, choices=WeldQualityModel.QUALITY_CHOICES)
    confidence = models.FloatField()
    inspection_date = models.DateTimeField(auto_now_add=True)
    model_used = models.ForeignKey(WeldQualityModel, on_delete=models.CASCADE)
    
    def __str__(self):
        return f"Inspection {self.id} - {self.prediction} ({self.confidence:.2f})" 