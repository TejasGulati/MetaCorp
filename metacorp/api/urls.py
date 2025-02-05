from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CompanyViewSet, ScenarioViewSet, SimulationViewSet

router = DefaultRouter()
router.register(r'companies', CompanyViewSet)
router.register(r'scenarios', ScenarioViewSet)
router.register(r'simulations', SimulationViewSet)

urlpatterns = [
    path('', include(router.urls)),
]