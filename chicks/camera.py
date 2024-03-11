from ultralytics import YOLO, checks, hub
checks()

hub.login('3df551543c4f102c16f0c425cf8e45b272879e5a25')

model = YOLO('https://hub.ultralytics.com/models/7yjB9ELCULTlw7ZEZziS')
results = model(source=0, show=True, save=True)