from rest_framework import serializers
from core.models import JointType, CalibrationSession, SensorData, CalibrationModel

class JointTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = JointType
        fields = '__all__'

class SensorDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SensorData
        fields = '__all__'

class CalibrationModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = CalibrationModel
        fields = '__all__'

class CalibrationSessionSerializer(serializers.ModelSerializer):
    sensor_data = SensorDataSerializer(many=True, read_only=True)
    models = CalibrationModelSerializer(many=True, read_only=True)

    class Meta:
        model = CalibrationSession
        fields = '__all__' 