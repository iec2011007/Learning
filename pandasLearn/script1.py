import pandas as pd
import datetime
import numpy as np

#create a list of datetime objects from a start date to end date

dt = datetime.datetime(2013,12,1)
end=datetime.datetime(2013,12,8)
step = datetime.timedelta(days=1)

dates= []

while(dt < end) :
    dates.append(dt.strftime('%m-%d'))
    dt += step

print dates


