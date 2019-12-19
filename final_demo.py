import argparse
import json
from ScrapyCensys import domain2ip
from ScrapyCensys import fetch_json
from app1_demo.XGB_lowdim import demo1_lowdim, demo1_highdim
from app2_low.demo import demo2_low, demo2_high


def main(args):
    ip = args.ip
    content_dict = []
    if args.load_json:
        # content_dict = json.load(open(r"C:\Users\11818\Desktop\RL\Code\vae\data.json"))
        with open(r"C:\Users\11818\Desktop\RL\Code\vae\data.json") as f:
            for line in f.readlines():
                content_dict.append(json.loads(line))
    else:
        if args.search_input == "ip":
            if ip == None:
                try:
                    ip = domain2ip(args.url)
                except:
                    print("url error")
                    return
            content_dict.append(fetch_json(ip, store=False, url_direct=False))
        else:
            content_dict.append(fetch_json(args.url, store=False, url_direct=True))
    print("url is {} ip address is {}".format(args.url, ip))
    if "demo1" in args.demo:
        demo1_highdim(content_dict)
        demo1_lowdim(content_dict)
    elif "demo2" in args.demo:
        demo2_high(content_dict)
        demo2_low(content_dict)
    else:
        print("no such demo")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--search_input", default="url", type=str)
    parser.add_argument("--url", default="baidu.com", type=str)
    parser.add_argument("--ip", default="172.105.237.241", type=str)
    parser.add_argument("--demo", default="demo1", type=str)
    parser.add_argument("--load_json", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
