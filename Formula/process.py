import logging
from multiprocessing import Process, current_process
import queue
from importlib import reload
import numpy as np
import pandas as pd
import fastf1 as ff1
from tqdm import tqdm


def get_logger():
    reload(logging)
    log = logging.getLogger("threading_example")
    log.setLevel(logging.DEBUG)

    fh = logging.StreamHandler()
    fmt = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


def get_driver_ahead_speed(data, driver_row, pbar) -> np.int_:
    driver_ahead_num = driver_row['DriverAhead']
    pbar.update()
    if str(driver_ahead_num) == '':
        return np.NaN
    driver_ahead_data = data[data['DriverNumber'] == driver_ahead_num]
    driver_ahead_data = driver_ahead_data[(driver_ahead_data['Time'] >= driver_row['Time']) &
                                          (driver_ahead_data['Time'] < driver_row['Time'] + pd.Timedelta("1s")) &
                                          (driver_ahead_data['X_sector'] == driver_ahead_data['X_sector']) &
                                          (driver_ahead_data['Y_sector'] == driver_ahead_data['Y_sector'])]
    return driver_ahead_data.iloc[0]['Speed']


def process_driver(data, driver_num, log) -> pd.DataFrame:
    driver_data = data[data['DriverNumber'] == driver_num].copy()
    driver_data['X_sector_diff'] = driver_data['X_sector'].diff()
    driver_data['Y_sector_diff'] = driver_data['Y_sector'].diff()
    driver_data = driver_data[(driver_data['X_sector_diff'] != 0) | (driver_data['Y_sector_diff'] != 0)]
    log.debug(f'started processing for {driver_num}')
    tqdm.pandas()
    pbar = tqdm()
    driver_data['Ahead_driver_speed'] = driver_data.progress_apply(lambda d: get_driver_ahead_speed(data, d, pbar), axis=1)
    log.debug(f'processing by: {current_process().name} ended')
    driver_data.drop(columns=['X_sector_diff', 'Y_sector_diff'], inplace=True)
    return driver_data


def form_overall_df(data, drivers_list, x_sector_length, y_sector_length) -> pd.DataFrame:
    return_data = data.pick_driver(drivers_list[0]).get_telemetry()
    return_data['DriverNumber'] = [drivers_list[0]] * len(return_data)
    for driver in drivers_list[1:]:
        driver_data = data.pick_driver(driver).get_telemetry()
        driver_data['DriverNumber'] = [driver] * len(driver_data)
        return_data = return_data.append(driver_data)

    return_data['X_sector'] = return_data['X'] // x_sector_length
    return_data['Y_sector'] = return_data['Y'] // y_sector_length
    return return_data


if __name__ == '__main__':
    x_size_of_sector = 50
    y_size_of_sector = 50

    ff1.Cache.enable_cache('cache')
    session = ff1.get_session(2021, 20, 'R')
    laps = session.load_laps(with_telemetry=True)
    session.load_telemetry()
    logger = get_logger()
    drivers = list(laps['DriverNumber'].unique())
    logger.debug(f'{len(drivers)} to process')
    all_drivers_data = form_overall_df(laps, drivers, x_size_of_sector, y_size_of_sector)
    drivers_sectors = process_driver(all_drivers_data, drivers[0], logger)
    drivers = drivers[1:]
    threads = list()
    que = queue.Queue()

    i = 0
    NUM_OF_THREADS = 2
    while i < len(drivers):
        for j in range(NUM_OF_THREADS):
            thread = Process(
                target=process_driver,
                name=drivers[i],
                args=(all_drivers_data.copy(), drivers[i], logger))
            thread.start()
            threads.append(thread)
            i += 1

        for j in range(len(threads)):
            threads[j].join()

        while not que.empty():
            result = que.get()
            drivers_sectors.append(result)

    # for i in range(len(drivers)):
    #     result = process_driver(all_drivers_data, drivers[i], logger)
    #     drivers_sectors.append(result)

    drivers_sectors.to_csv("sectors.csv")
