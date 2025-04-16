from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from core.models import JointType, CalibrationSession, SensorData, CalibrationModel
from .serializers import (
    JointTypeSerializer, CalibrationSessionSerializer,
    SensorDataSerializer, CalibrationModelSerializer
)
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

class JointTypeViewSet(viewsets.ModelViewSet):
    queryset = JointType.objects.all()
    serializer_class = JointTypeSerializer
    filter_backends = [DjangoFilterBackend]

class CalibrationSessionViewSet(viewsets.ModelViewSet):
    queryset = CalibrationSession.objects.all()
    serializer_class = CalibrationSessionSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['joint_type', 'is_completed']

    @action(detail=True, methods=['post'])
    def complete_session(self, request, pk=None):
        session = self.get_object()
        session.is_completed = True
        session.save()
        return Response({'status': 'session completed'})

class SensorDataViewSet(viewsets.ModelViewSet):
    queryset = SensorData.objects.all()
    serializer_class = SensorDataSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['session', 'material_type']

class CalibrationModelViewSet(viewsets.ModelViewSet):
    queryset = CalibrationModel.objects.all()
    serializer_class = CalibrationModelSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['session', 'model_type']

    @action(detail=True, methods=['post'])
    def predict(self, request, pk=None):
        model = self.get_object()
        try:
            loaded_model = joblib.load(model.model_file)
            input_data = request.data.get('input_data', [])
            if not input_data:
                return Response(
                    {'error': 'No input data provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            predictions = loaded_model.predict(np.array(input_data))
            return Response({'predictions': predictions.tolist()})
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) 