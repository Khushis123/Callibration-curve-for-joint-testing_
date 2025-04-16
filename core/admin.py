from django.contrib import admin
from .models import JointType, CalibrationSession, SensorData, CalibrationModel

@admin.register(JointType)
class JointTypeAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at', 'updated_at')
    search_fields = ('name',)

@admin.register(CalibrationSession)
class CalibrationSessionAdmin(admin.ModelAdmin):
    list_display = ('joint_type', 'operator', 'date_time', 'created_at')
    list_filter = ('joint_type', 'operator')
    search_fields = ('joint_type__name', 'operator')

@admin.register(SensorData)
class SensorDataAdmin(admin.ModelAdmin):
    list_display = ('session', 'applied_load', 'sensor_voltage', 'temperature', 'humidity', 'machine_speed', 'material_type')
    list_filter = ('session', 'material_type')
    search_fields = ('session__joint_type__name', 'material_type')

@admin.register(CalibrationModel)
class CalibrationModelAdmin(admin.ModelAdmin):
    list_display = ('session', 'model_type', 'accuracy', 'r2_score', 'mae', 'training_date')
    list_filter = ('model_type', 'session__joint_type')
    search_fields = ('session__joint_type__name', 'model_type') 