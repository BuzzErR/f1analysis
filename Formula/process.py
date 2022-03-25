import datetime
import logging
from multiprocessing import Process, current_process
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
    fh = logging.FileHandler("processing.log")
    fmt = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


def get_driver_ahead_data(data, driver_data):
    driver_ahead_data_to_return = {"Driver_ahead_speed": [], "Driver_ahead_DRS": [], "Driver_ahead_throttle": [],
                                   "Overtake": []}
    columns_of_ahead = ["Driver_ahead_speed", "Driver_ahead_DRS", "Driver_ahead_throttle", "Overtake"]
    driver_ahead_data_to_return = pd.DataFrame(driver_ahead_data_to_return)
    driver_data = driver_data.reset_index(drop=True)
    current = current_process()
    for index, driver_row in tqdm(driver_data.iloc[:-1].iterrows(), total=driver_data.shape[0] - 1,
                                  position=current._identity[0] - 1, desc=f'Driver: {current.name}'):
        driver_ahead_num = driver_row['DriverAhead']
        driver_ahead_data = data[data['DriverNumber'] == driver_ahead_num]
        ahead_row = driver_ahead_data[(driver_ahead_data['Time'] >= driver_row['Time']) &
                                      (driver_ahead_data['Time'] < driver_row['Time'] + pd.Timedelta("0.5s")) &
                                      (driver_ahead_data['X_sector'] == driver_row['X_sector']) &
                                      (driver_ahead_data['Y_sector'] == driver_row['Y_sector'])].sort_values(
            by=['Time'])
        if str(driver_ahead_num) == '' or len(ahead_row) == 0:
            data_to_append = np.empty(4)
            data_to_append[:] = np.nan
            data_to_append[3] = 0
        else:
            ahead_driver_row = ahead_row.iloc[0][['Speed', 'DRS', 'Throttle']].to_list()
            overtake_completed = 0
            next_row = driver_data.iloc[index + 1]
            if next_row['DriverAhead'] != driver_ahead_num:
                next_sector_tick_driver_ahead = driver_ahead_data[
                    (driver_ahead_data['Time'] >= next_row['Time'] - pd.Timedelta("0.5s")) &
                    (driver_ahead_data['Time'] < next_row['Time'] + pd.Timedelta("0.5s")) &
                    (driver_ahead_data['X_sector'] == next_row['X_sector']) &
                    (driver_ahead_data['Y_sector'] == next_row['Y_sector'])].sort_values(by=['Time'])
                if len(next_sector_tick_driver_ahead) > 0 and \
                        next_sector_tick_driver_ahead.iloc[0]['Time'] > next_row['Time']:
                    overtake_completed = 1
            ahead_driver_row.append(overtake_completed)
            data_to_append = ahead_driver_row
        driver_ahead_data_to_return.append(pd.DataFrame([data_to_append], columns=columns_of_ahead))
    return driver_ahead_data_to_return


def process_driver(data, driver_num):
    log = get_logger()
    driver_data = data[data['DriverNumber'] == driver_num].copy()
    driver_data['X_sector_diff'] = driver_data['X_sector'].diff()
    driver_data['Y_sector_diff'] = driver_data['Y_sector'].diff()
    driver_data = driver_data[(driver_data['X_sector_diff'] != 0) | (driver_data['Y_sector_diff'] != 0)]
    start_time = datetime.datetime.now()
    log.info(f'started processing for {driver_num}')
    current = current_process()
    driver_ahead_data = get_driver_ahead_data(data, driver_data)
    driver_data = pd.concat([driver_data, driver_ahead_data], axis=1)
    log.info(f'processing by: {current.name} ended, took {datetime.datetime.now() - start_time}')
    driver_data.drop(columns=['X_sector_diff', 'Y_sector_diff'], inplace=True)
    driver_data.to_csv(f"{current.name}.csv")
    return


def form_overall_df(data, drivers_list, x_sector_length, y_sector_length) -> pd.DataFrame:
    if len(drivers_list) == 0:
        return pd.DataFrame({})
    return_data = data.pick_driver(drivers_list[0]).get_telemetry()
    return_data['DriverNumber'] = [drivers_list[0]] * len(return_data)
    for driver_to_process in drivers_list[1:]:
        driver_data = data.pick_driver(driver_to_process).get_telemetry()
        driver_data['DriverNumber'] = [driver_to_process] * len(driver_data)
        return_data = return_data.append(driver_data)

    return_data['X_sector'] = return_data['X'] // x_sector_length
    return_data['Y_sector'] = return_data['Y'] // y_sector_length
    return return_data


def form_general_df(drivers_list, log):
    result = pd.read_csv(f"{drivers_list[0]}.csv")
    for driver_to_process in drivers_list[1:]:
        data = pd.read_csv(f"{driver_to_process}.csv")
        result = result.append(data)
    log.debug(f'{len(result)} total len of dataframe')
    return result


def get_drivers_to_process(drivers_list, log):
    result = []
    return_drivers = []
    for driver_to_process in drivers_list:
        try:
            df = pd.read_csv(f'{driver_to_process}.csv')
            log.debug(f'{len(df)} rows added')
            result.append(df)
        except Exception as _:
            return_drivers.append(driver_to_process)
    if len(result) > 0:
        df = result[0]
        for data in result[1:]:
            df = df.append(data)

    return return_drivers, result


if __name__ == '__main__':

    X_SIZE_OF_SECTOR = 100
    Y_SIZE_OF_SECTOR = 100
    NUM_OF_THREADS = 4

    ff1.Cache.enable_cache('cache')
    session = ff1.get_session(2021, 20, 'R')
    laps = session.load_laps(with_telemetry=True)
    session.load_telemetry()
    logger = get_logger()

    drivers, preloaded_data = get_drivers_to_process(list(laps['DriverNumber'].unique()), logger)
    logger.info(f'{len(drivers)} to process')
    all_drivers_data = form_overall_df(laps, drivers, X_SIZE_OF_SECTOR, Y_SIZE_OF_SECTOR)
    planed_threads = list()
    active_threads = list()

    for driver in drivers:
        thread = Process(
            target=process_driver,
            name=driver,
            args=(all_drivers_data.copy(), driver))
        planed_threads.append(thread)

    i = 0
    while i < len(drivers) or len(active_threads) > 0:
        while len(active_threads) < NUM_OF_THREADS and len(planed_threads) > 0:
            thread = planed_threads[-1]
            active_threads.append(thread)
            active_threads[-1].start()
            planed_threads = planed_threads[:-1]
            i += 1

        logger.info(f'{len(active_threads)} processes in active pool')
        active_threads = [thread for thread in active_threads if thread.is_alive()]
        logger.info(f'{len(active_threads)} processes left, {len(planed_threads)} more to process')

        time.sleep(5)

    overall_df = form_general_df(list(laps['DriverNumber'].unique()), logger)
    overall_df.to_csv(f"complete_df.csv")
