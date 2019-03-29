# coding=utf-8
from datetime import datetime


def parser(x):
    return datetime.strptime(x, "%Y-%m-%d")
    #return datetime.strptime(x, "%m/%d/%Y")