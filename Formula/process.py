import logging
from multiprocessing import Process, current_process, Queue
import multiprocessing
from importlib import reload
import numpy as np
import pandas as pd
import fastf1 as ff1
from tqdm import tqdm
import time


def get_logger():
    reload(logging)
    log = logging.getLogger("threading_example")
    log.setLevel(logging.DEBUG)

    # fh = logging.StreamHandler()
    fh = logging.FileHandler("processing.log", mode='w')
    fmt = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


def get_driver_ahead_speed(data, driver_row, pbar) -> np.int_:
    driver_ahead_num = driver_row['DriverAhead']
    pbar.update(1)
    if str(driver_ahead_num) == '':
        return np.NaN
    driver_ahead_data = data[data['DriverNumber'] == driver_ahead_num]
    driver_ahead_data = driver_ahead_data[(driver_ahead_data['Time'] >= driver_row['Time']) &
                                          (driver_ahead_data['Time'] < driver_row['Time'] + pd.Timedelta("1s")) &
                                          (driver_ahead_data['X_sector'] == driver_ahead_data['X_sector']) &
                                          (driver_ahead_data['Y_sector'] == driver_ahead_data['Y_sector'])]
    if len(driver_ahead_data) == 0:
        return np.NaN
    return driver_ahead_data.iloc[0]['Speed']


def process_driver(data, driver_num, log, lock):
    driver_data = data[data['DriverNumber'] == driver_num].copy()
    driver_data['X_sector_diff'] = driver_data['X_sector'].diff()
    driver_data['Y_sector_diff'] = driver_data['Y_sector'].diff()
    driver_data = driver_data[(driver_data['X_sector_diff'] != 0) | (driver_data['Y_sector_diff'] != 0)]
    log.debug(f'started processing for {driver_num}')
    tqdm.pandas()
    current = current_process()
    with lock:
        bar = tqdm(
            desc=f'Driver: {current.name}',
            total=len(driver_data),
            position=current._identity[0] - 1,
            leave=False
        )
    driver_data['Ahead_driver_speed'] = driver_data.apply(lambda d: get_driver_ahead_speed(data, d, bar), axis=1)
    log.debug(f'processing by: {current.name} ended')
    driver_data.drop(columns=['X_sector_diff', 'Y_sector_diff'], inplace=True)
    with lock:
        bar.close()
    driver_data.to_csv(f"{current.name}.csv")


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


def form_general_df(drivers_list):
    result = pd.read_csv(f"{drivers_list[0]}.csv")
    for driver in drivers_list[1:]:
        result.append(pd.read_csv(f"{driver}.csv"))
    return result


if __name__ == '__main__':

    X_SIZE_OF_SECTOR = 50
    Y_SIZE_OF_SECTOR = 50
    NUM_OF_THREADS = 4

    lock_mp = multiprocessing.Manager().Lock()
    ff1.Cache.enable_cache('cache')
    session = ff1.get_session(2021, 20, 'R')
    laps = session.load_laps(with_telemetry=True)
    session.load_telemetry()
    logger = get_logger()

    drivers = list(laps['DriverNumber'].unique())
    logger.debug(f'{len(drivers)} to process')
    all_drivers_data = form_overall_df(laps, drivers, X_SIZE_OF_SECTOR, Y_SIZE_OF_SECTOR)

    planed_threads = list()
    active_threads = list()

    for driver in drivers:
        thread = Process(
            target=process_driver,
            name=driver,
            args=(all_drivers_data.copy(), driver, logger, lock_mp))
        planed_threads.append(thread)

    i = 0
    while i < len(drivers) or len(active_threads) > 0:
        if len(active_threads) < NUM_OF_THREADS and len(planed_threads) > 0:
            thread = planed_threads[-1]
            active_threads.append(thread)
            active_threads[-1].start()
            planed_threads = planed_threads[:-1]
            i += 1

        for j in range(len(active_threads)):
            if not active_threads[j].is_alive():
                active_threads = active_threads.pop(j)

        time.sleep(3)

    overall_df = form_general_df(drivers)
    overall_df.to_csv(f"complete_df.csv")
