"""
	* @author [cb]
	* @email [cbqin0323@gmail.com]
	* @create date 2018-05-07 11:07:09
	* @modify date 2018-05-07 11:07:09
	* @desc [description]
"""

import os
import json


def reload(paths):
    for path in paths:
        with open(path, 'r') as f:
            context = json.load(f)
        with open(path.replace('cmrc2018_', ''), 'w') as f:
            json.dump(context, f, ensure_ascii=False)


def main():
    prefix = '../data/raw/cmrc2018/'
    paths = [os.path.join(prefix, file) for file in os.listdir(prefix)]
    reload(paths)


if __name__ == '__main__':
    main()
