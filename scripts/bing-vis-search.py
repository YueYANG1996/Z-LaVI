#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os 
import requests
import argparse
import hashlib
import imghdr
import os
import pickle
import posixpath
import re
import signal
import socket
import threading
import time
import urllib.parse
import urllib.request


'''
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = ''
endpoint = 'https://api.bing.microsoft.com/v7.0/images/search'#os.environ['BING_SEARCH_V7_ENDPOINT'] + "/bing/v7.0/search"

# Query term(s) to search for. 

# Construct a request
mkt = 'en-US'
headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

def get_links(noun, offset):
    params = { 'q': noun, 'mkt': mkt, 'offset':offset}
    # Call the API
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        links = []
        for r in response.json()['value']:
            links.append(r['contentUrl'])
    except Exception as ex:
        raise ex
    return links

# config
output_dir = './bing'  # default output dir
socket.setdefaulttimeout(2)

tried_urls = []
image_md5s = {}
in_progress = 0
urlopenheader = {'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0'}

def download(pool_sema: threading.Semaphore, img_sema: threading.Semaphore, url: str, output_dir: str, limit: int, keyword: str):
    global in_progress

    if url in tried_urls:
        print('SKIP: Already checked url, skipping')
    pool_sema.acquire()
    in_progress += 1
    acquired_img_sema = False
    path = urllib.parse.urlsplit(url).path
    filename = posixpath.basename(path).split('?')[0]  # Strip GET parameters from filename
    name, ext = os.path.splitext(filename)
    name = name[:36].strip()
    filename = name + ext

    try:
        request = urllib.request.Request(url, None, urlopenheader)
        image = urllib.request.urlopen(request).read()
        if not imghdr.what(None, image):
            print('SKIP: Invalid image, not saving ' + filename)
            return

        md5_key = hashlib.md5(image).hexdigest()
        if md5_key in image_md5s:
            print('SKIP: Image is a duplicate of ' + image_md5s[md5_key] + ', not saving ' + filename)
            return

        i = 0
        while os.path.exists(os.path.join(output_dir, filename)):
            if hashlib.md5(open(os.path.join(output_dir, '{}_{}_{}'.format(keyword, i, )), 'rb').read()).hexdigest() == md5_key:
                print('SKIP: Already downloaded ' + filename + ', not saving')
                return
            i += 1
            filename = "%s-%d%s" % (name, i, ext)

        image_md5s[md5_key] = filename

        img_sema.acquire()
        acquired_img_sema = True
        # if limit is not None and len(tried_urls) >= limit:
        #     return

        imagefile = open(os.path.join(output_dir, filename), 'wb')
        imagefile.write(image)
        imagefile.close()
        print(" OK : " + filename)
        tried_urls.append(url)
    except Exception as e:
        print("FAIL: " + filename)
    finally:
        pool_sema.release()
        if acquired_img_sema:
            img_sema.release()
        in_progress -= 1


def fetch_images_from_keyword(pool_sema: threading.Semaphore, img_sema: threading.Semaphore, keyword: str,
                              output_dir: str, filters: str, limit: int):
    current = 0
    last = ''
    offset = 0
    while True:
        time.sleep(0.1)

        if in_progress > 10:
            continue

        if offset > limit:
            return
        links = get_links(keyword, offset)
        offset += 35
        try:
            if links[-1] == last:
                return
            for index, link in enumerate(links[:limit]):
                # if limit is not None and len(tried_urls) >= limit:
                #     exit(0)
                t = threading.Thread(target=download, args=(pool_sema, img_sema, link, output_dir, limit,keyword))
                t.start()
                current += 1
            last = links[-1]
        except IndexError:
            print('FAIL: No search results for "{0}"'.format(keyword))
            return

def backup_history(*args):
    download_history = open(os.path.join(output_dir, 'download_history.pickle'), 'wb')
    pickle.dump(tried_urls, download_history)
    copied_image_md5s = dict(
        image_md5s)  # We are working with the copy, because length of input variable for pickle must not be changed during dumping
    pickle.dump(copied_image_md5s, download_history)
    download_history.close()
    print('history_dumped')
    if args:
        exit(0)

def download_images(query, number_of_images, output_dir, filters=''):
    output_sub_dir = os.path.join(output_dir_origin, keyword.strip().replace(' ', '_'))
    if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)
        
    fetch_images_from_keyword(pool_sema, img_sema, query, output_sub_dir, filters, number_of_images)
    # time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bing image bulk downloader')
    parser.add_argument('-s', '--search-string', help='Keyword to search', required=False)
    parser.add_argument('-f', '--search-file', help='Path to a file containing search strings line by line',
                        required=False)
    parser.add_argument('-o', '--output', help='Output directory', required=False)
    parser.add_argument('--adult-filter-off', help='Disable adult filter', action='store_true', required=False)
    parser.add_argument('--filters',
                        help='Any query based filters you want to append when searching for images, e.g. +filterui:license-L1',
                        required=False)
    parser.add_argument('--limit', help='Make sure not to search for more than specified amount of images.', type=int, default=30)
    parser.add_argument('--threads', help='Number of threads', type=int, default=200)
    args = parser.parse_args()
    start = time.time()
    if (not args.search_string) and (not args.search_file):
        parser.error('Provide Either search string or path to file containing search strings')
    if args.output:
        output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir_origin = output_dir
    signal.signal(signal.SIGINT, backup_history)
    try:
        download_history = open(os.path.join(output_dir, 'download_history.pickle'), 'rb')
        tried_urls = pickle.load(download_history)
        image_md5s = pickle.load(download_history)
        download_history.close()
    except (OSError, IOError):
        tried_urls = []
    if args.adult_filter_off:
        urlopenheader['Cookie'] = 'SRCHHPGUSR=ADLT=OFF'
    pool_sema = threading.BoundedSemaphore(args.threads)
    img_sema = threading.Semaphore()
    if args.search_string:
        fetch_images_from_keyword(pool_sema, img_sema, args.search_string, output_dir, args.filters, args.limit)
    elif args.search_file:
        try:
            inputFile = open(args.search_file)
        except (OSError, IOError):
            print("FAIL: Couldn't open file {}".format(args.search_file))
            exit(1)
       
        for keyword in inputFile.readlines():
            keyword = keyword.strip()
            output_sub_dir = os.path.join(output_dir_origin, keyword.strip().replace(' ', '_').replace('/', '_'))
            if not os.path.exists(output_sub_dir):
                os.makedirs(output_sub_dir)
            fetch_images_from_keyword(pool_sema, img_sema, keyword, output_sub_dir, args.filters, args.limit)
            # backup_history()
            # time.sleep(1)
        inputFile.close()
    print(time.time()- start)
