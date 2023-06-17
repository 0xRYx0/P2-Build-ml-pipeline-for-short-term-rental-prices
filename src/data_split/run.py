#!/usr/bin/env python
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity     :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project     :  2 - Build an ML Pipeline for Short-term Rental Prices in NYC
  Step        :  Initial Training: Data Split (data_split)
  Description :  Extract and segregate test set
  Author      :  Rakan Yamani
  Date        :  15 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import argparse
import logging
import wandb
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split


logging.basicConfig(
    filename='./../logs/logging.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s - %(message)s',
    datefmt="%m/%d/%y %I:%M:%S %p")
logger = logging.getLogger()
logger.info("SUCCESS: accessing logging.log file")

# refrenced Udacity repo
def go(args):

    run = wandb.init(job_type="data_split")
    run.config.update(args)
    
    artifact_local_path = run.use_artifact(args.input_data).file()
    logging.info(f"SUCCESS: Downloading artifact {args.input_data}")
    
    df = pd.read_csv(artifact_local_path)
    
    logger.info("Splitting the dataset")
    train_val, test = train_test_split(df, test_size=args.test_size, random_state=args.random_state,
                                       stratify=df[args.stratify] if args.stratify != "none" else None)
    
    for df, split in zip([train_val, test], ["trainval", "test"]):
        
        with tempfile.NamedTemporaryFile("w") as fp:            
            df.to_csv(fp.name, index=False)
            
            artifact = wandb.Artifact(
                name=f"{split}_data.csv",
                type=f"{split}_data",
                description=f"{split} split of dataset {args.input_data}",
            )
            
            artifact.add_file(fp.name)
            
            run.log_artifact(artifact)
            logger.info(f"SUCCESS: Logging artifact {split}_data.csv dataset")
            
            artifact.wait()
            
        logging.info(f"SUCCESS: Uploading the {split}_data.csv dataset")




if __name__ == "__main__":

    # running main script:
    
    parser = argparse.ArgumentParser(description="To extract and segregate test set")

    parser.add_argument(
        "--input_data", 
        type=str,
        help='Input dataset to perform the required splitting',
        required=True
        )

    parser.add_argument(
        "--test_size", 
        type=float,
        help='Assigned size for testset',
        required=True
    )

    parser.add_argument(
        "--random_state", 
        type=int,
        help='Value used as a seed to generate random numbers (Usually fixed for reproducibility)',
        required=False
    )

    parser.add_argument(
        "--stratify", 
        type=str,
        help='train_test_split method will return training and test subsets that have the same proportions of class labels as the input dataset',
        required=False
    )

    args = parser.parse_args()

    go(args)
