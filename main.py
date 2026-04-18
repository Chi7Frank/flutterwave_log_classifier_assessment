import os

print("StartUp Pipeline")
os.system("python src/preprocess.py")
os.system("python src/train.py")
os.system("python src/evaluate.py")
os.system("python src/inference.py")
print("Pipeline Finished. Check outputs/ folder for metrics and predictions.")