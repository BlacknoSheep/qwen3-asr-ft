import datetime

SAMPLE_RATE = 16000


def get_utc_time_str(format="%Y-%m-%d_%H-%M-%S", timedelta_hours: int = 8):
    utc = datetime.timezone(datetime.timedelta(hours=timedelta_hours))
    current_time = datetime.datetime.now(utc)
    time_str = current_time.strftime(format)
    return time_str
