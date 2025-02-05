from django.db import models

class Company(models.Model):
    name = models.CharField(max_length=200)
    industry = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class Scenario(models.Model):
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='scenarios')
    name = models.CharField(max_length=200)
    description = models.TextField()
    parameters = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.company.name} - {self.name}"


class Simulation(models.Model):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    )
    
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, related_name='simulations')
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='pending')
    results = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"Simulation for {self.scenario.name}"