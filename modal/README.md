# Import the class into the app

```python
Pi3Remote = modal.Cls.from_name(“pi3-inference”, "ModelInference")
```

# Instantiate the class

```python
model = ModelInference()
result_path = model.run_inference.remote("/data/auditorio/iphone13")
```