from django.db import models
from django.db.models import JSONField

class SimulationResult(models.Model):
    input_data = models.JSONField()
    results = models.JSONField()
    insights = models.JSONField()
    visualizations = models.JSONField(null=True, blank=True)  # Add this field
    is_parallel = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),  # Added index for better performance
        ]

    def __str__(self):
        sim_type = "Parallel" if self.is_parallel else "Single"
        return f"{sim_type} Simulation - {self.created_at.strftime('%Y-%m-%d %H:%M')}"