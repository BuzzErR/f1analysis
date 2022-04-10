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


def get_logger(year_number, race, tqdm_enable):
    reload(logging)
    log = logging.getLogger("threading_example")
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f"log/processing_{year_number}_{race}.log")
    if tqdm_enable:
        fh = logging.StreamHandler()
    fmt = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


def get_driver_rows_to_process(driver_data, current, tqdm_enable):
    return tqdm(driver_data.iloc[:-1].iterrows(), total=driver_data.shape[0] - 1,
                position=current._identity[0] - 1,
                desc=f'Driver: {current.name}') \
        if tqdm_enable else driver_data.iloc[:-1].iterrows()


def get_driver_ahead_data(data, driver_data, tqdm_enable):
    driver_ahead_data_to_return = {"Driver_ahead_speed": [], "Driver_ahead_DRS": [], "Driver_ahead_throttle": [],
                                   "Overtake": []}
    columns_of_ahead = ["Driver_ahead_speed", "Driver_ahead_DRS", "Driver_ahead_throttle", "Overtake"]
    driver_ahead_data_to_return = pd.DataFrame(driver_ahead_data_to_return)
    driver_data = driver_data.reset_index(drop=True)
    current = current_process()
    for index, driver_row in get_driver_rows_to_process(driver_data, current, tqdm_enable):
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


def process_driver(data, driver_num, year_number, race_number, time_limit, tqdm_enable):
    log = get_logger(year_number, race_number, tqdm_enable)
    # нужно, чтобы отсечь эту "секунду" после, которая может помешать при анализе
    driver_data = data[(data['DriverNumber'] == driver_num) & (data['Time'] < time_limit)].copy()
    driver_data['X_sector_diff'] = driver_data['X_sector'].diff()
    driver_data['Y_sector_diff'] = driver_data['Y_sector'].diff()
    driver_data = driver_data[(driver_data['X_sector_diff'] != 0) | (driver_data['Y_sector_diff'] != 0)]
    start_time = datetime.datetime.now()
    log.info(f'started processing for {driver_num}')
    current = current_process()
    driver_ahead_data = get_driver_ahead_data(data, driver_data, tqdm_enable)
    driver_ahead_data.reset_index(inplace=True, drop=True)
    driver_data.reset_index(inplace=True, drop=True)
    driver_data = pd.concat((driver_data, driver_ahead_data), axis=1)
    log.info(f'processing by: {current.name} ended, took {datetime.datetime.now() - start_time}, '
             f'{time_limit} limit processed, {len(driver_data)} rows')
    driver_data.drop(columns=['X_sector_diff', 'Y_sector_diff'], inplace=True)
    driver_data = driver_data[driver_data['Driver_ahead_speed'].notna()]
    driver_data = driver_data.drop_duplicates()
    if os.path.exists(get_driver_csv_directory(year_number, race_number, current.name)):
        driver_data.to_csv(get_driver_csv_directory(year_number, race_number, current.name), mode='a', index=False,
                           header=False)
    else:
        driver_data.to_csv(get_driver_csv_directory(year_number, race_number, current.name), index=False)
    return


def form_overall_df(data, drivers_list, x_sector_length, y_sector_length, time_limit_min, time_limit_max):
    if len(drivers_list) == 0:
        return pd.DataFrame({}), None
    # прибавляем секнду, чтобы адекватно находить записи гонщиков впереди
    time_limit_max += pd.to_timedelta('0 days 00:00:01')
    try:
        return_data = data.pick_driver(drivers_list[0]).get_telemetry()
    except ValueError as e:
        return pd.DataFrame({}), e
    return_data['DriverNumber'] = [drivers_list[0]] * len(return_data)
    return_data = return_data[(time_limit_min < return_data['Time']) & (return_data['Time'] < time_limit_max)]
    for driver_to_process in drivers_list[1:]:
        driver_data = data.pick_driver(driver_to_process).get_telemetry()
        driver_data = driver_data[(time_limit_min < driver_data['Time']) & (driver_data['Time'] < time_limit_max)]
        driver_data['DriverNumber'] = [driver_to_process] * len(driver_data)
        return_data = pd.concat((return_data, driver_data))

    return_data['X_sector'] = return_data['X'] // x_sector_length
    return_data['Y_sector'] = return_data['Y'] // y_sector_length
    return return_data, None


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
    return result.drop_duplicates()


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


def parse_list(input_string):
    input_string = input_string.replace(' ', '')
    if input_string[0] != '{' and input_string[-1] != '}':
        return [int(input_string)]

    input_string = input_string[1:-1]
    if input_string.find('-') != -1:
        input_string = list(map(int, input_string.split('-')))
        return list(range(input_string[0], input_string[-1]))
    else:
        return list(map(int, input_string.split('-')))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Attention! You'll see multiple progress bars."
                                                 "\nHow to use this program:")
    group_races = parser.add_mutually_exclusive_group(required=True)
    group_years = parser.add_mutually_exclusive_group(required=True)

    group_races.add_argument("-r", nargs='+', type=int,
                             help="Number of race to process or you can select multiple races like this:"
                                  "1 2 4... Note that if you have chosen multiple years race numbers will be"
                                  "processed for each")
    group_races.add_argument("--range_races", nargs='+', type=int,
                             help="Range of races to process:"
                                  "1 4... Note that if you have chosen multiple years race numbers will be"
                                  "processed for each")
    group_years.add_argument("-y", nargs='+', type=int,
                             help="Year to process or you can select multiple years like this:"
                                  "2019 2020 2021...")
    group_years.add_argument("--range_years", nargs='+', type=int, help="Range of years to process")

    parser.add_argument("-d", help="Time delta in minutes", required=True)
    parser.add_argument("-s", nargs='?', help="Process everything without tqdm progress bar")
    args = parser.parse_args()
    years = args.y
    if args.range_years is not None:
        years = list(range(args.range_years[0], args.range_years[-1] + 1))

    races = args.r
    if args.range_races is not None:
        races = list(range(args.range_races[0], args.range_races[-1] + 1))
    return years, races, int(args.d), args.s is None


def create_directory(year_number, race_number):
    exists = os.path.exists(f"races/{year_number}")
    if not exists:
        os.makedirs(f"races/{year_number}")
    exists = os.path.exists(f"races/{year_number}/{race_number}")
    if not exists:
        os.makedirs(f"races/{year_number}/{race_number}")


def get_driver_csv_directory(year_number, race_number, driver):
    return f"races/{year_number}/{race_number}/{driver}.csv"


def main():
    X_SIZE_OF_SECTOR = 100
    Y_SIZE_OF_SECTOR = 100
    NUM_OF_THREADS = 4

    years_to_process, races_to_process, time_delta_minutes, enable_terminal = parse_arguments()
    time_delta_hours = time_delta_minutes // 60
    time_delta_minutes = time_delta_minutes % 60
    print(years_to_process, races_to_process)
    for year_number in years_to_process:
        for race_number in races_to_process:
            # setup raw data, directories, logger
            ff1.Cache.enable_cache('cache')
            session = ff1.get_session(year_number, race_number, 'R')
            laps = session.load_laps(with_telemetry=True)
            session.load_telemetry()
            logger = get_logger(year_number, race_number, enable_terminal)
            logger.info(f"{year_number} out of: {years_to_process}, {race_number} race, out of {races_to_process}")
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
                all_drivers_data, error = form_overall_df(laps, all_drivers, X_SIZE_OF_SECTOR, Y_SIZE_OF_SECTOR, start_time,
                                                   start_time +
                                                   delta)
                if error is not None:
                    logger.critical(error)
                    continue
                start_time += delta
                planed_threads = list()
                active_threads = list()

                for driver in drivers:
                    thread = Process(
                        target=process_driver,
                        name=driver,
                        args=(all_drivers_data.copy(), driver, year_number, race_number, start_time, enable_terminal))
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
                    overall_df.to_csv(get_driver_csv_directory(year_number, race_number, "overall"), mode='a',
                                      index=False,
                                      header=False)
                else:
                    overall_df.to_csv(get_driver_csv_directory(year_number, race_number, "overall"))


if __name__ == '__main__':
    main()
