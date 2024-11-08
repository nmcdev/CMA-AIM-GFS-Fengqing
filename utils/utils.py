from datetime import datetime, timedelta

def day_of_year(year, month, day):
    # 构造日期对象
    date_object = datetime.strptime(f'{year}/{month}/{day}', '%Y/%m/%d')
    # 计算一年中的第几天
    day_of_year = date_object.timetuple().tm_yday
    if not ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0):
        if month >= 3:
            day_of_year += 1 
        
    return day_of_year