import datetime
import logging
from multiprocessing import Process, current_process
from importlib import reload
import numpy as np
import pandas as pd
import fastf1 as ff1
import pandas.errors
from tqdm import tqdm
import time
import os
import argparse


def get_logger(year_number, race):
    reload(logging)
    log = logging.getLogger("threading_example")
    log.setLevel(logging.DEBUG)

    # fh = logging.StreamHandler()
    fh = logging.FileHandler(f"processing_{year_number}_{race}.log")
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
                                  position=current._identity[0] - 1,
                                  desc=f'Driver: {current.name}'):
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
        data_to_append = pd.DataFrame([data_to_append], columns=columns_of_ahead)
        driver_ahead_data_to_return.reset_index(inplace=True, drop=True)
        driver_ahead_data_to_return = pd.concat((driver_ahead_data_to_return, data_to_append))

    # т.к. ранее обработали на одну строку меньше
    data_to_append = np.empty(4)
    data_to_append[:] = np.nan
    data_to_append[3] = 0
    data_to_append = pd.DataFrame([data_to_append], columns=columns_of_ahead)
    driver_ahead_data_to_return = pd.concat((driver_ahead_data_to_return, data_to_append))

    return driver_ahead_data_to_return


def process_driver(data, driver_num, year_number, race_number, time_limit):
    log = get_logger(year_number, race_number)
    # нужно, чтобы отсечь эту "секунду" после, которая может помешать при анализе
    driver_data = data[(data['DriverNumber'] == driver_num) & (data['Time'] < time_limit)].copy()
    driver_data['X_sector_diff'] = driver_data['X_sector'].diff()
    driver_data['Y_sector_diff'] = driver_data['Y_sector'].diff()
    driver_data = driver_data[(driver_data['X_sector_diff'] != 0) | (driver_data['Y_sector_diff'] != 0)]
    start_time = datetime.datetime.now()
    log.info(f'started processing for {driver_num}')
    current = current_process()
    driver_ahead_data = get_driver_ahead_data(data, driver_data)
    driver_ahead_data.reset_index(inplace=True, drop=True)
    driver_data.reset_index(inplace=True, drop=True)
    driver_data = pd.concat((driver_data, driver_ahead_data), axis=1)
    log.info(f'processing by: {current.name} ended, took {datetime.datetime.now() - start_time}')
    driver_data.drop(columns=['X_sector_diff', 'Y_sector_diff'], inplace=True)
    driver_data = driver_data[driver_data['Driver_ahead_speed'].notna()]

    if os.path.exists(get_driver_csv_directory(year_number, race_number, current.name)):
        driver_data.to_csv(get_driver_csv_directory(year_number, race_number, current.name), mode='a', index=False,
                           header=False)
    else:
        driver_data.to_csv(get_driver_csv_directory(year_number, race_number, current.name), index=False)
    return


def form_overall_df(data, drivers_list, x_sector_length, y_sector_length, time_limit_min, time_limit_max) -> \
        pd.DataFrame:
    if len(drivers_list) == 0:
        return pd.DataFrame({})

    # прибавляем секнду, чтобы адекватно находить записи гонщиков впереди
    time_limit_max += pd.to_timedelta('0 days 00:00:01')

    return_data = data.pick_driver(drivers_list[0]).get_telemetry()
    return_data['DriverNumber'] = [drivers_list[0]] * len(return_data)
    return_data = return_data[(time_limit_min < return_data['Time']) & (return_data['Time'] < time_limit_max)]
    for driver_to_process in drivers_list[1:]:
        driver_data = data.pick_driver(driver_to_process).get_telemetry()
        driver_data = driver_data[(time_limit_min < driver_data['Time']) & (driver_data['Time'] < time_limit_max)]
        driver_data['DriverNumber'] = [driver_to_process] * len(driver_data)
        return_data = pd.concat((return_data, driver_data))

    return_data['X_sector'] = return_data['X'] // x_sector_length
    return_data['Y_sector'] = return_data['Y'] // y_sector_length
    return return_data


def form_general_df(drivers_list, year_number, race_number, log):
    first_driver = 0
    while True:
        try:
            result = pd.read_csv(get_driver_csv_directory(year_number, race_number, drivers_list[first_driver]))
            break
        except pandas.errors.EmptyDataError:
            first_driver += 1

    for driver_to_process in drivers_list[first_driver + 1:]:
        try:
            data = pd.read_csv(get_driver_csv_directory(year_number, race_number, driver_to_process))
            result = pd.concat([result, data], axis=0)
        except pandas.errors.EmptyDataError:
            pass
    log.debug(f'{len(result)} total len of dataframe')
    return result.drop_duplicates


def get_drivers_to_process(drivers_list, log, year_number, race):
    result = []
    return_drivers = []
    for driver_to_process in drivers_list:
        try:
            df = pd.read_csv(get_driver_csv_directory(year_number, race, driver_to_process))
            log.debug(f'{len(df)} rows added')
            result.append(df)
        except FileNotFoundError as _:
            return_drivers.append(driver_to_process)
        except pandas.errors.EmptyDataError as _:
            pass

    if len(result) > 0:
        df = result[0]
        for data in result[1:]:
            df = pd.concat((df, data))

    return return_drivers, result


def parse_arguments():
    parser = argparse.ArgumentParser(description="Attention! You'll see multiple progress bars."
                                                 "\nHow to use this program:")
    parser.add_argument("-r", help="Number of race to process", required=True)
    parser.add_argument("-y", help="Year to process", required=True)
    parser.add_argument("-d", help="Time delta in minutes", required=True)
    args = parser.parse_args()
    return args.y, args.r, int(args.d)


def create_directory(year_number, race_number):
    isExist = os.path.exists(f"races/{year_number}")
    if not isExist:
        os.makedirs(f"races/{year_number}")
    isExist = os.path.exists(f"races/{year_number}/{race_number}")
    if not isExist:
        os.makedirs(f"races/{year_number}/{race_number}")


def get_driver_csv_directory(year_number, race_number, driver):
    return f"races/{year_number}/{race_number}/{driver}.csv"


def main():
    X_SIZE_OF_SECTOR = 100
    Y_SIZE_OF_SECTOR = 100
    NUM_OF_THREADS = 4

    year_number, race_number, time_delta_minutes = parse_arguments()
    time_delta_hours = time_delta_minutes // 60
    time_delta_minutes = time_delta_minutes % 60

    # setup raw data, directories, logger
    ff1.Cache.enable_cache('cache')
    session = ff1.get_session(year_number, race_number, 'R')
    laps = session.load_laps(with_telemetry=True)
    session.load_telemetry()
    logger = get_logger(year_number, race_number)
    create_directory(year_number, race_number)

    all_drivers = list(laps['DriverNumber'].unique())
    drivers, preloaded_data = get_drivers_to_process(all_drivers, logger, year_number, race_number)
    logger.info(f'{len(drivers)} to process')
    if len(drivers) == 0:
        overall_df = form_general_df(all_drivers, year_number, race_number, logger)
        overall_df.to_csv(get_driver_csv_directory(year_number, race_number, "overall"))
        return

    start_time = pd.to_timedelta('0 days 00:00:00')
    laps['LapEndTime'] = laps['LapStartTime'] + laps['LapTime']
    latest_time = laps['LapEndTime'].max()
    while start_time <= latest_time:
        delta = pd.to_timedelta(f'0 days {time_delta_hours}:{time_delta_minutes}:00')
        all_drivers_data = form_overall_df(laps, all_drivers, X_SIZE_OF_SECTOR, Y_SIZE_OF_SECTOR, start_time, start_time +
                                           delta)
        start_time += delta
        planed_threads = list()
        active_threads = list()

        for driver in drivers:
            thread = Process(
                target=process_driver,
                name=driver,
                args=(all_drivers_data.copy(), driver, year_number, race_number, start_time))
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
        overall_df = form_general_df(all_drivers, year_number, race_number, logger)
        if os.path.exists(get_driver_csv_directory(year_number, race_number, "overall")):
            overall_df.to_csv(get_driver_csv_directory(year_number, race_number, "overall"), mode='a', index=False,
                               header=False)
        else:
            overall_df.to_csv(get_driver_csv_directory(year_number, race_number, "overall"))


if __name__ == '__main__':
    main()
