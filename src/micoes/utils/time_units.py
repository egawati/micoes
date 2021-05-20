import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def time_unit_numpy(time_unit):
    time_npunit = None
    if time_unit == 'seconds':
        time_npunit = "s"
    elif time_unit == 'milliseconds':
        time_npunit = "ms"
    elif time_unit == 'microseconds':
        time_npunit = "us"
    elif time_unit == 'nanoseconds':
        time_npunit = "ns"
    elif time_unit == 'picoseconds':
        time_npunit = "ps"
    elif time_unit == 'femtoseconds':
        time_npunit = "fs"
    elif time_unit == 'attoseconds':
        time_npunit = "as"
    elif time_unit == 'minutes':
        time_npunit = "m"
    elif time_unit == 'hours':
        time_npunit = "h"
    elif time_unit == "days":
        time_npunit = "D"
    elif time_unit == 'weeks':
        time_npunit = "W"
    elif time_unit == "months":
        time_npunit = "M"
    elif time_unit == "years":
        time_npunit = "Y"
    else:
        logging.info(f('Undefined time unit {time_unit}'))
    return time_npunit
