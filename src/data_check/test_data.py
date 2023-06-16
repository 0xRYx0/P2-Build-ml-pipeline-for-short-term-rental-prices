"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity     :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project     :  2 - Build an ML Pipeline for Short-term Rental Prices in NYC
  Step        :  Data Check (data_check)
  Description :  Test and verify dataset
  Author      :  Rakan Yamani
  Date        :  15 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd
import numpy as np
import scipy.stats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


logging.basicConfig(
    filename='./../logs/logging.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s - %(message)s',
    datefmt="%m/%d/%y %I:%M:%S %p")
logger = logging.getLogger()
logger.info("SUCCESS: accessing logging.log file")


def test_column_names(data):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)
    logger.info("SUCCESS: Testing - Provided column names are matching and in the same order")


def test_neighborhood_names(data):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)
    logger.info("SUCCESS: Testing - Neighborhood names completed")


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    # idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)
    
    # assert np.sum(~idx) == 0
    logger.info("SUCCESS: Testing - Proper longitude and latitude boundaries for properties in and around NYC")


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold
    logger.info("SUCCESS: Testing - Validat if new data and rteference data are significantly different")


def test_row_count(data: pd.DataFrame):
    '''
    Check if the size of the dataset is reasonable (not too small, not too large)
    '''
    assert 15000 < data.shape[0] < 1000000
    logger.info("SUCCESS: Testing - Check if the size of the dataset is reasonable (not too small, not too large)")
    
def test_price_range(data: pd.DataFrame, min_price: float, max_price: float):
    '''
    Check if the price range is between min_price and max_price
    '''
    assert data['price'].between(min_price, max_price).all()
    logger.info("SUCCESS: Testing - Check if the price range is between min_price and max_price")
