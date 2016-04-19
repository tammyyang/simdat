#!/usr/bin/env python3

'''
This is a simple script to download images from Bing.
One can also replace BingImageCrawler by other crawler easily,
just go to https://pypi.python.org/pypi/icrawler/0.1.5
and you should find easily how to achieve it.

Usage:
    0. Install icrawler
        $pip3 install icrawler
    1. Create download_args.json to contain keywords you want to use:
        {
         "search_for": {
                        "apple": ["fruit", "pie"],
                        "banana": []
                       }
        }
    2. Execute the script
       $./image_downloader.py -n 10 -o 100

For more details, use ./image_downloader.py --help

'''

import time
import os
import sys
import json
import argparse
from icrawler.examples import BingImageCrawler


def parse_json(fname):
    """Parse the input profile

    @param fname: input profile path
    @return data: a dictionary with user-defined data for training

    """
    with open(fname) as data_file:
        data = json.load(data_file)
    return data

data = parse_json('download_args.json')
search_for = data['search_for']


def check_dir(dirpath):
    """Check if a directory exists, create one elsewise."""
    if not os.path.exists(dirpath):
        print("Creating %s" % dirpath)
        os.makedirs(dirpath)


def main():
    parser = argparse.ArgumentParser(
        description="Simple tool to download images via Google Search."
        )
    parser.add_argument(
        "-t", "--test", action='store_true'
        )
    parser.add_argument(
        "-p", "--path", type=str, default=os.getcwd(),
        help="Path to output images"
        )
    parser.add_argument(
        "-n", "--num", type=int, default=100,
        help="How many images do you need."
        )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of threads to run."
        )
    parser.add_argument(
        "-o", "--offset", type=int, default=0,
        help="Offset."
        )
    parser.add_argument(
        "--min", type=int, default=None,
        help="Minimum size of the image."
        )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Maximum size of the image."
        )

    args = parser.parse_args()

    t0 = time.time()
    check_dir(args.path)
    for kwd in search_for:
        subdir = os.path.join(args.path, kwd)
        check_dir(subdir)
        print (" Item name = ", kwd)
        if len(search_for[kwd]) == 0:
            bing_crawler = BingImageCrawler(subdir)
            bing_crawler.crawl(
                keyword=kwd, offset=args.offset, max_num=args.num,
                feeder_thr_num=1, parser_thr_num=1,
                downloader_thr_num=args.threads,
                min_size=args.min, max_size=args.max)
        else:
            for j in range(0, len(search_for[kwd])):
                print("    : %s" % search_for[kwd][j])
                ssubdir = os.path.join(subdir, search_for[kwd][j])
                check_dir(ssubdir)
                pure_keyword = '%20' + search_for[kwd][j]
                pure_keyword = kwd + pure_keyword.replace(' ', '%20')
                bing_crawler = BingImageCrawler(ssubdir)
                bing_crawler.crawl(
                    keyword=pure_keyword, offset=args.offset,
                    max_num=args.num, feeder_thr_num=1,
                    parser_thr_num=1, downloader_thr_num=args.threads,
                    min_size=args.min, max_size=args.max)

if __name__ == '__main__':
    main()
