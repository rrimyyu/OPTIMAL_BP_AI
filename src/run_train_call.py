# run_train_call.py
from run_train import main

if __name__ == "__main__":
    main([
        "--excel-path", r"C:/Users/ORI1/Desktop/RY/RESEARCH_PHD/DATA/OPTIMAL-BP_AI/Core - 20230728 OPTIMAL-BP - nam 42.xlsx",
        "--results-dir", "results",
        "--seed", "42",
        "--test-size", "0.3",
        "--epochs", "300",
        "--batch-size", "64",
        "--class-weight", "auto",
        "--use-pp",
    ])
