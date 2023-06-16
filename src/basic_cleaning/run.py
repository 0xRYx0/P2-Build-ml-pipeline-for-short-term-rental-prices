#!/usr/bin/env python
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity     :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project     :  2 - Build an ML Pipeline for Short-term Rental Prices in NYC
  Step        :  Basic Cleaning (basic_cleaning)
  Description :  Download from W&B the raw dataset and apply some basic 
                 data cleaning, exporting the result to a new artifact
  Author      :  Rakan Yamani
  Date        :  14 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import os
import wandb
import logging
import argparse
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


logging.basicConfig(
    filename='./../logs/logging.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s - %(message)s',
    datefmt="%m/%d/%y %I:%M:%S %p")
logger = logging.getLogger()
logger.info("SUCCESS: Creating logging file named 'logging.log'")


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)
    
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    logger.info("SUCCESS: Downloading artifact from W&B o start the clean-up process")
    
    
    df = pd.read_csv(artifact_path)   
    logger.info("SUCCESS: Loading artifact to dataframe")
    
    
    df['last_review'] = pd.to_datetime(df['last_review'])
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    print("SUCCESS: Completing data clean-up process") 
    
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("SUCCESS: Applied required fixes") 
    
    filename = "clean_sample.csv"
    df.to_csv(filename, index=False)
    
    artifact = wandb.Artifact(
        name=args.output_artifact_name,
        type=args.output_artifact_type,
        description=args.output_artifact_description,
    )
    logger.info("SUCCESS: Creating artifact to be uploaded to W&B")
    
    artifact.add_file(filename)
    logger.info("SUCCESS: Adding artifact to W&B")

    run.log_artifact(artifact)
    logger.info("SUCCESS: Logging artifact to W&B")
    
    os.remove(filename)
    logger.info("SUCCESS: Removing procesing file")


if __name__ == "__main__":

    # running main script
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The name used for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact_name", 
        type=str,
        help="The name used for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact_type", 
        type=str,
        help="The type assigned for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact_description", 
        type=str,
        help="The description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price value for a given unit",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price value for a given unit",
        required=True
    )

    args = parser.parse_args()
    
    logger.info("SUCCESS: Preparing parser arguments")
    
    go(args)
    
    logger.info("SUCCESS: Completed the script, well done")
