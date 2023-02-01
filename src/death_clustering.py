import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import Models
from src import Evaluate
from src import DataImport
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import matplotlib.gridspec as gridspec

from matplotlib.gridspec import SubplotSpec



domains = DataImport.domain_dict()
df_by_us = DataImport.data_reader_by_us(bio=False)

id_columns = ['hhid','pn','hhidpn','interview_year']

